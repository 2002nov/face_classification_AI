import torch
from ultralytics import YOLO
import cv2
import numpy as np

# โหลดโมเดลที่เทรนแล้ว
model = YOLO('Clsweights/weights1/best.pt')  # แทนที่ด้วย path ของโมเดลที่ใช้จริง

# โหลดรูปที่ต้องการทดสอบ
input_img_path = 'FakeImage1/Real-2.png'  # แทนที่ด้วย path ของรูปที่ต้องการทดสอบ
input_img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)

# รันการคาดการณ์
results = model(input_img_path)

# ตรวจสอบว่าผลลัพธ์มีความน่าจะเป็นของ class
if hasattr(results[0], 'probs') and results[0].probs is not None:
    # เข้าถึงความน่าจะเป็นของ class และรับ class ที่คาดการณ์ได้
    probs = results[0].probs.data  # ใช้ .data เพื่อเข้าถึง tensor ภายใน
    predicted_class = torch.argmax(probs).item()  # รับค่า index ของความน่าจะเป็นที่สูงที่สุด

    # ตรวจสอบว่าคลาสที่คาดการณ์ได้คือ class1 (สมมติว่า class1 มี label เป็น 0)
    if predicted_class == 0:
        # ในกรณีนี้เราจะคำนวณความแตกต่างระหว่างรูปที่ทดสอบกับการคาดการณ์ของโมเดลโดยตรง
        predicted_img = results[0].orig_img

        # ปรับขนาดรูปที่ใส่เข้าไปให้มีขนาดเท่ากับรูปที่โมเดลคาดการณ์ได้
        input_img_resized = cv2.resize(input_img, (predicted_img.shape[1], predicted_img.shape[0]))

        # ตรวจสอบและแปลงภาพเป็นสีเทา (ถ้าจำเป็น)
        if len(input_img_resized.shape) == 3 and input_img_resized.shape[2] == 3:
            input_img_resized = cv2.cvtColor(input_img_resized, cv2.COLOR_BGR2GRAY)
        if len(predicted_img.shape) == 3 and predicted_img.shape[2] == 3:
            predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2GRAY)

        # คำนวณความแตกต่าง
        diff = cv2.absdiff(input_img_resized, predicted_img)
        diff_percentage = np.sum(diff) / (diff.size * 255) * 100

        print(f"เปอร์เซ็นต์ความแตกต่างระหว่างรูปที่ใส่เข้าไปกับรูปที่โมเดลคาดการณ์: {diff_percentage:.2f}%")
    else:
        print("รูปที่ใส่เข้าไปไม่ได้ถูกคาดการณ์เป็น class1")
else:
    print("ผลลัพธ์ไม่มีข้อมูลความน่าจะเป็นของ class")