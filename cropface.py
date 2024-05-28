import cv2
import os
from ultralytics import YOLO

# โหลดโมเดลที่เทรนแล้ว
model = YOLO(r"C:\Users\Keen\Desktop\YOLOv8\facedetect/yolov8l-face.pt")

# ฟังก์ชันสำหรับตรวจจับใบหน้าด้วย YOLOv8
def face_detector(image):
    results = model.predict(source=image)
    face_locations = []
    for result in results:
        for box in result.boxes.xyxy:
            left, top, right, bottom = map(int, box)
            face_locations.append((top, right, bottom, left))
    return face_locations

# ตั้งค่าโฟลเดอร์ที่มีรูปภาพหลายๆ รูป 
image_folder = r'C:\Users\Keen\Desktop\YOLOv8\facedetect\testface'
output_folder = r'C:\Users\Keen\Desktop\YOLOv8\facedetect\result3'

# สร้างโฟลเดอร์สำหรับบันทึกภาพใบหน้าที่ตัดออกมา หากยังไม่มี
os.makedirs(output_folder, exist_ok=True)

# วนลูปผ่านไฟล์ในโฟลเดอร์
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        # ใช้โมเดล YOLOv8 ในการค้นหาตำแหน่งของใบหน้าในภาพ
        face_locations = face_detector(image)

        # สร้างตัวนับเพื่อให้ชื่อไฟล์ของภาพใบหน้าที่ถูกตัดออกมาไม่ซ้ำกัน
        face_counter = 0

        # ตัดเฉพาะส่วนของใบหน้าที่ถูกตรวจจับได้และบันทึกเป็นภาพใหม่
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            face_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_face_{face_counter}.jpg")
            
            # บันทึกภาพใบหน้าที่ถูกตัดออกมา
            cv2.imwrite(face_filename, face_image)
            face_counter += 1

        print(f"{face_counter} faces detected and saved from {filename}.")
