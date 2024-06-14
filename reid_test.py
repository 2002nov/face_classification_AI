import cv2 as cv
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# Load YOLOv8 model for detection
detect_model = YOLO('/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/Work_Main/yolov8l-pose.pt')

# Load Re-ID model
def load_custom_model(model_path, num_classes=751, use_cuda=True):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if use_cuda and torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

# Preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom Re-ID model path
custom_model_path = '/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/ReID/models/best150.pt'
use_cuda = torch.cuda.is_available()
reid_model = load_custom_model(custom_model_path, use_cuda=use_cuda)
reid_model.fc = nn.Sequential()
reid_model = reid_model.eval()
if use_cuda:
    reid_model = reid_model.cuda()

# Function to crop and extract features
def crop_and_extract_features(frame, bboxes, reid_model, transform, use_cuda=True):
    features = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cropped_image = frame[y1:y2, x1:x2]
        cropped_pil_image = Image.fromarray(cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB))
        image_tensor = transform(cropped_pil_image).unsqueeze(0)
        if use_cuda:
            image_tensor = image_tensor.cuda()
        
        with torch.no_grad():
            feature = reid_model(image_tensor).cpu().numpy()
        features.append(feature)
    
    return features

# Similarity calculation function
def calculate_similarity(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)

# Assign IDs based on feature similarity
def assign_ids(features_frame1, features_frame2):
    cost_matrix = np.zeros((len(features_frame1), len(features_frame2)))
    for i, feature1 in enumerate(features_frame1):
        for j, feature2 in enumerate(features_frame2):
            cost_matrix[i, j] = calculate_similarity(feature1, feature2)
    
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    assignments = {}
    for row, col in zip(row_indices, col_indices):
        assignments[col] = row
    return assignments

# Process the video and perform Re-ID with pose tracking
def process_video(video_path, detect_model, reid_model, output_dir, frame_interval=30, use_cuda=True):
    cap = cv.VideoCapture(video_path)
    frame_count = 0
    prev_features = []
    prev_bboxes = []
    id_counter = 0
    id_map = {}

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if frame_count % frame_interval == 0:
            results = detect_model(frame)

            bboxes = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bboxes.append((x1, y1, x2, y2))
            
            features = crop_and_extract_features(frame, bboxes, reid_model, data_transforms, use_cuda=use_cuda)
            
            if prev_features:
                assignments = assign_ids(prev_features, features)
                new_id_map = {}
                for idx, bbox in enumerate(bboxes):
                    if idx in assignments and assignments[idx] in id_map:
                        assigned_id = id_map[assignments[idx]]
                    else:
                        assigned_id = id_counter
                        id_counter += 1
                    new_id_map[idx] = assigned_id
                    person_label = f'Person {assigned_id + 1}'
                    cv.putText(frame, person_label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                id_map = new_id_map
            else:
                for idx, bbox in enumerate(bboxes):
                    assigned_id = id_counter
                    id_map[idx] = id_counter
                    person_label = f'Person {id_counter + 1}'
                    cv.putText(frame, person_label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    id_counter += 1

            prev_features = features
            prev_bboxes = bboxes
        else:
            for idx, bbox in enumerate(bboxes):
                if idx in id_map:
                    assigned_id = id_map[idx]
                    person_label = f'Person {assigned_id + 1}'
                    cv.putText(frame, person_label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        frame_count += 1
        cv.imshow('frame', frame)
        
        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

# Example usage
video_path = '/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/Work_Main/TestVideo/CCTV.mp4'
#video_path = '/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/Work_Main/TestVideo/Scale_test5.mp4'
output_dir = '/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/ReID/reid_output'
process_video(video_path, detect_model, reid_model, output_dir, use_cuda=use_cuda)

