import pathlib
# import YOLO model
from ultralytics import YOLO

# Load a model
model = YOLO("runs/classify/train6/weights/best.pt") # load a pretrained model (recommended for training)

class_dict = {0: 'Angry', 1: 'Bored', 2: 'Confused', 3: 'cool', 4: 'Errrr', 5: 'Funny', 6: 'Happy', 7: 'Normal', 8: 'Proud', 9: 'Sad', 10: 'Scared', 11: 'Shy', 12: 'Sigh', 13: 'super_angry', 14: 'Surprised', 15: 'Suspicious', 16: 'sweet', 17: 'tricky', 18: 'unhappy', 19: 'Worried'}

class_predict = []

files = [f for f in pathlib.Path().glob("testface/*.png")]
files.sort(key=lambda x: int(x.stem))  # เรียงลำดับตามตัวเลขที่อยู่ในชื่อไฟล์
for i in files:
    path = str(i)
    results = model.predict(path)
    class_predict.append(class_dict[results[0].probs.data.argmax().item()])

print(class_predict)
