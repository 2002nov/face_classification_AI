from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
from ultralytics import YOLO

def search_file(image_name):
    for root, dirs, files in os.walk("Real12Amulate"):
        for file in files:
            if file == image_name:
                return os.path.join(root, file)
    return None

def browse_file():
    app = QApplication(sys.argv)
    file_path, _ = QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
    app.quit()
    return file_path

def detect_objects(image_path):
    # Load the YOLOv8 segmentation model
    model = YOLO('weights/best.pt')

    # Define names in Thai
    names = [
        "พระผงวัดปากน้ำ", "หลวงปู๋ทวดปี24", "พระขุนแผน", "เหรียญหลวงปู่โต๊ะ", "เหรียญหลวงปู่พ่อคุณ",
        "พระชัยวัฒน์", "พระเศียร์โล้นสะดุ้งกลับปรกโพธิ์", "หลวงพ่อโสธรสองหน้า", "หลวงปู่ทวดพิมพ์เตารีดใหญ่",
        "เหรียญหลวงพ่อคูณเนื้อทองแดง", "เหรียญหลวงพ่อคูณเนื้อเงิน", "พระกิ่งพระพุทธยอดฟ้า"
    ]

    # Load the image
    img = cv2.imread(image_path)
    
    # Perform inference
    results = model(img)

    # Convert image to OpenCV format for drawing
    img_with_boxes = img.copy()
    
    cls = " - "  # Initialize class index
    conf = 0.00  # Initialize confidence

    # Draw bounding boxes and labels on the image
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                conf = box.conf.item()
                cls = int(box.cls.item())
                label = f"{names[cls]} {conf:.2f}"
                
                # Calculate bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Convert the frame to PIL image
                pil_img = Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                
                # Calculate text position to be centered at the top edge of the bounding box
                font_path = 'Sarabun/Sarabun-Thin.ttf'
                font = ImageFont.truetype(font_path, 14)
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x1 + (x2 - x1) // 2 - text_width // 2
                text_y = y1 - text_height - 10
                
                # Show the confidence and class name at the center of the top edge
                draw.text((text_x, text_y), label, font=font, fill=(255, 0, 0))
                
                # Convert the PIL image back to OpenCV format
                img_with_boxes = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                # Draw bounding box
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            mask_combined = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            for mask in masks:
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_combined = np.maximum(mask_combined, mask_resized)

            # Convert mask_combined to 8-bit
            mask_combined = mask_combined.astype(np.uint8)

            # Create a color map for the mask
            colored_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            colored_mask[mask_combined > 0] = [0, 255, 0]  # Example color: green

            # Blend the original frame with the colored mask
            img_with_boxes = cv2.addWeighted(img_with_boxes, 1.0, colored_mask, 0.5, 0)
    
    if cls is None or conf is None:
        return "", 0.0, img_with_boxes
    
    return names[cls], conf, img_with_boxes
