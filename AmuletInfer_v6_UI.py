import cv2
import cvui
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from AmuletInfer_v6_Model import browse_file, detect_objects, search_file

WINDOW_NAME = 'CVUI Amulet'

# Define the size of the window
window_height = 680  # Increased height
window_width = 800  # Increased width

# Define the size of the black box
box_height = 300
box_width = 300
box_x = 40
box_y = 110

# Define the size of the right black box
right_box_x = window_width - box_x - box_width

# Text box position and size
label_x = box_x
textbox_x = box_x + 50
textbox_y = box_height + 130
textbox_width = box_width - textbox_x
textbox_height = 30
textbox_spacing = 50

# Set the initial position for the buttons and checkbox
button_x = box_x + 10
button_spacing = 10
button_y = textbox_y + 170
button_width = 120
button_height = 40

# Load a font that supports Thai
font_path = 'Sarabun/Sarabun-Thin.ttf'  # replace with the path to a Thai font
font = ImageFont.truetype(font_path, 14)

# Create a black canvas
canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)

# Create white boxes at the top left and right corners
canvas[box_y:box_y + box_height, box_x:box_x + box_width] = (50, 50, 50)
canvas[box_y:box_y + box_height, right_box_x:right_box_x + box_width] = (50, 50, 50)

# Create a window
cv2.namedWindow(WINDOW_NAME)
cvui.init(WINDOW_NAME)

cls_name = " - "
conf = 0.00
selected_file_path = None
searched_file_path = None

while True:
    # Render buttons
    if cvui.button(canvas, button_x, button_y, button_width, button_height, 'Browse'):
        print('Browsing file')
        selected_file_path = browse_file()
        print(f"Selected file: {selected_file_path}")

    if cvui.button(canvas, button_x + button_width + button_spacing, button_y, button_width, button_height, 'Search'):
        print('Amulet is searched')
        if selected_file_path:
            searched_file_path = search_file(f"{cls_name}.jpg")
            print(f"Searched file: {searched_file_path}")

    if cvui.button(canvas, window_width - button_width - 20, 20, button_width, button_height, 'Close'):
        print('Window is closing')
        break
    
    # If a file is selected, detect objects and display the image with bounding boxes on the left canvas
    if selected_file_path:
        cls_name, conf, img_with_boxes = detect_objects(selected_file_path)
        if cls_name is not None and conf is not None:
            # Display the image with bounding boxes
            if img_with_boxes.shape[1] > box_width or img_with_boxes.shape[0] > box_height:
                aspect_ratio = img_with_boxes.shape[1] / img_with_boxes.shape[0]
                if img_with_boxes.shape[1] > img_with_boxes.shape[0]:
                    new_width = box_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = box_height
                    new_width = int(new_height * aspect_ratio)
                img_with_boxes_resized = cv2.resize(img_with_boxes, (new_width, new_height))
            else:
                img_with_boxes_resized = img_with_boxes

            y_offset = box_y + (box_height - img_with_boxes_resized.shape[0]) // 2
            x_offset = box_x + (box_width - img_with_boxes_resized.shape[1]) // 2

            canvas[box_y:box_y + box_height, box_x:box_x + box_width] = (50, 50, 50)  # Reset the area to white
            canvas[y_offset:y_offset + img_with_boxes_resized.shape[0], x_offset:x_offset + img_with_boxes_resized.shape[1]] = img_with_boxes_resized

    # If a file is searched, display the image on the right canvas
    if searched_file_path:
        searched_img = cv2.imread(searched_file_path)
        if searched_img is not None:
            if searched_img.shape[1] > box_width or searched_img.shape[0] > box_height:
                aspect_ratio = searched_img.shape[1] / searched_img.shape[0]
                if searched_img.shape[1] > searched_img.shape[0]:
                    new_width = box_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = box_height
                    new_width = int(new_height * aspect_ratio)
                searched_img_resized = cv2.resize(searched_img, (new_width, new_height))
            else:
                searched_img_resized = searched_img

            y_offset = box_y + (box_height - searched_img_resized.shape[0]) // 2
            x_offset = right_box_x + (box_width - searched_img_resized.shape[1]) // 2

            canvas[box_y:box_y + box_height, right_box_x:right_box_x + box_width] = (50, 50, 50)  # Reset the area to white
            canvas[y_offset:y_offset + searched_img_resized.shape[0], x_offset:x_offset + searched_img_resized.shape[1]] = searched_img_resized

    # Convert the canvas to PIL image for text rendering
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Render the text for the real image and browse image
    draw.text((window_width//2 + box_x + box_width//2, box_y - 40), "พระแท้", font=font, fill=(255, 255, 255))
    draw.text((box_width//2 - box_x//2, box_y - 40), "พระที่ต้องการทดสอบ", font=font, fill=(255, 255, 255))

    # Render the Label text box
    draw.text((label_x, textbox_y + 5), "Label: ", font=font, fill=(255, 255, 255))
    draw.rectangle([textbox_x, textbox_y, textbox_x + textbox_width, textbox_y + textbox_height], fill=(50, 50, 50))
    draw.text((textbox_x + 5, textbox_y + 5), cls_name, font=font, fill=(255, 255, 255))

    # Render the Confidence text box
    draw.text((label_x, textbox_y + 5 + textbox_spacing), "Score: ", font=font, fill=(255, 255, 255))
    draw.rectangle([textbox_x, textbox_y + textbox_spacing, textbox_x + textbox_width, textbox_y + textbox_height + textbox_spacing], fill=(50, 50, 50))
    draw.text((textbox_x + 10, textbox_y + 5 + textbox_spacing), f"{conf:.2f}", font=font, fill=(255, 255, 255))

    # Render the Variant text box
    draw.text((label_x, textbox_y + 5 + 2 * textbox_spacing), "Variant: ", font=font, fill=(255, 255, 255))
    draw.rectangle([textbox_x, textbox_y + 2 * textbox_spacing, textbox_x + textbox_width, textbox_y + textbox_height + 2 * textbox_spacing], fill=(50, 50, 50))
    draw.text((textbox_x + 10, textbox_y + 5 + 2 * textbox_spacing), "variant", font=font, fill=(255, 255, 255))

    # Render the ความเหมือนพระแท้ text box
    draw.text((window_width - box_x - box_width, textbox_y + 5), "ความเหมือนพระแท้: ", font=font, fill=(255, 255, 255))

    # Render the percent text box
    draw.rectangle([window_width - box_x - box_width, textbox_y + textbox_spacing, window_width - box_x - box_width + 50, textbox_y + textbox_height + textbox_spacing], fill=(50, 50, 50))
    draw.text((window_width - box_x - box_width + 60, textbox_y + textbox_spacing + 5), " เปอร์เซ็น", font=font, fill=(255, 255, 255))

    # Render the angle text box
    draw.rectangle([window_width - box_x - box_width, textbox_y + textbox_spacing * 2, window_width - box_x - box_width + 50, textbox_y + textbox_height + textbox_spacing * 2], fill=(50, 50, 50))
    draw.text((window_width - box_x - box_width + 60, textbox_y + textbox_spacing * 2 + 5), " องศาหมุนตาม", font=font, fill=(255, 255, 255))

    # Render the projection angle text box
    draw.rectangle([window_width - box_x - box_width, textbox_y + textbox_spacing * 3, window_width - box_x - box_width + 50, textbox_y + textbox_height + textbox_spacing * 3], fill=(50, 50, 50))
    draw.text((window_width - box_x - box_width + 60, textbox_y + textbox_spacing * 3 + 5), " องศาโปรเจคชัน", font=font, fill=(255, 255, 255))

    # Convert back to OpenCV format
    canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Update cvui internal stuff
    cvui.update()

    # Show everything on the screen
    cv2.imshow(WINDOW_NAME, canvas)

    # Break the loop when 'ESC' is pressed
    if cv2.waitKey(30) == 27:
        break

# Clean up
cv2.destroyAllWindows()
