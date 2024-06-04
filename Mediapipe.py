import cv2 as cv
from ultralytics import YOLO
import os
import mediapipe as mp
import numpy as np

# Load YOLOv8 model for face detection
detect_model = YOLO('/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/yolov8l-face.pt')

# Load YOLO model for classification
classify_model = YOLO('/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/runs/classify/train15#/weights/best.pt')

class_dict = {0: 'Angry', 1: 'Bored', 2: 'Confused', 3: 'Cool', 4: 'Errrr', 5: 'Funny', 6: 'Happy', 7: 'Normal', 8: 'Proud', 9: 'Sad', 10: 'Scared', 11: 'Shy', 12: 'Sigh', 13: 'Superangry', 14: 'Surprised', 15: 'Suspicious', 16: 'Unhappy', 17: 'Worried', 18: 'sweet', 19: 'tricky'}

class_predict = []

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Open video file
cap = cv.VideoCapture('/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/TestVideo/MultiPeople_Test1.mp4')

# Create a directory to save the detected faces
output_dir = '/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/detected_faces'
os.makedirs(output_dir, exist_ok=True)

frame_counter = 0

while cap.isOpened():
    face_counter = 0  # Counter for detected faces
    faces = []

    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Detect objects using YOLOv8
    results = detect_model(frame)
    
    # Iterate through the detected objects
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to int
            conf = box.conf[0]  # Confidence
            cls = box.cls[0]  # Class
            label = f"{detect_model.names[int(cls)]} {conf:.2f}"
            
            # Draw bounding box and label on the frame
            cv.rectangle(frame, (x1-10, y1-10), (x2+20, y2+20), (0, 255, 0), 2)
            cv.putText(frame, label, (x1, y1 - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Extract the detected face
            face = frame[y1:y2, x1:x2]
            face_counter += 1
            faces.append(face)

            # Convert the face region from BGR to RGB
            face_rgb = cv.cvtColor(face, cv.COLOR_BGR2RGB)

            # Process the face with MediaPipe face mesh
            result = face_mesh.process(face_rgb)

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    # Draw face landmarks on the face region
                    mp_drawing.draw_landmarks(
                        image=face,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)

            # Classify the face
            predicts = classify_model.predict(face)
            prediction_idx = class_dict[predicts[0].probs.data.argmax().item()]
            print('frame' + str(frame_counter) + ' face' + str(face_counter) + ' : ' + prediction_idx)
            prediction = class_dict[prediction_idx]

            class_predict.append(prediction_idx)

    # Display the frame with detected objects
    cv.imshow('frame', frame)
    frame_counter += 1
    
    # Check for 'q' key press to exit
    if cv.waitKey(1) == ord('q'):
        break

# Release video file and close windows
cap.release()
cv.destroyAllWindows()

