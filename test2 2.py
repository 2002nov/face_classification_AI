import cv2 as cv
from ultralytics import YOLO

# โหลดโมเดล YOLOv8 จากไฟล์ .pt
model = YOLO(r'C:\Users\Keen\Desktop\YOLOv8\facedetect\yolov8l-face.pt')

# เปิดไฟล์วิดีโอ
cap = cv.VideoCapture(r'C:\Users\Keen\Desktop\YOLOv8\facial_expression\videotest02.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    
    # ถ้าอ่าน frame ได้อย่างถูกต้อง ret จะเป็น True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # ใช้โมเดล YOLOv8 เพื่อทำการตรวจจับวัตถุในเฟรม
    results = model(frame)
    
    # วาดกรอบสี่เหลี่ยมล้อมรอบวัตถุที่ตรวจจับได้
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # การแปลงตำแหน่งเป็น int
            conf = box.conf[0]  # ความมั่นใจ
            cls = box.cls[0]  # ประเภทของคลาส
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # แสดงภาพที่มีการตรวจจับวัตถุ
    cv.imshow('frame', frame)
    
    # รอการกดคีย์บอร์ด ถ้ากด 'q' จะออกจากโปรแกรม
    if cv.waitKey(1) == ord('q'):
        break

# ปิดการใช้ไฟล์วิดีโอและหน้าต่างที่เปิด
cap.release()
cv.destroyAllWindows()
