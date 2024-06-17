import cv2 as cv
from ultralytics import YOLO
import pathlib
import os
from PIL import Image
import numpy as np

# Load YOLOv8 model
detect_model = YOLO('/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/Work_Main/yolov8l-face.pt')

# Open video file
cap = cv.VideoCapture('/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/Work_Main/TestVideo/Scale_test5.mp4')

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

    # Display the frame with detected objects
    cv.imshow('frame', frame)
    
    # Check for 'q' key press to exit
    if cv.waitKey(1) == ord('q'):
        break

# Release video file and close windows
cap.release()
cv.destroyAllWindows()

