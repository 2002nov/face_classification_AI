import sys
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import PySimpleGUI as sg
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# Load models
modelClassify = YOLO("/media/lab_brownien/data1/Work_Student2024_V2/AI_train/runs/classify/train15#/weights/best.pt")
modelDetection = YOLO("/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/yolov8l-face.pt")
modelPose = YOLO('/media/lab_brownien/data1/Work_Student2024_V2/AI_train/Tang/yolov8l-pose.pt')

class_dict = {0: 'Angry', 1: 'Bored', 2: 'Confused', 3: 'Cool', 4: 'Errrr', 5: 'Funny', 6: 'Happy', 7: 'Normal', 
              8: 'Proud', 9: 'Sad', 10: 'Scared', 11: 'Shy', 12: 'Sigh', 13: 'Superangry', 14: 'Surprised', 
              15: 'Suspicious', 16: 'Unhappy', 17: 'Worried', 18: 'sweet', 19: 'tricky'}

initial_joint_positions = [
    (20, 8, "Nose"), (0, 0, "Left Eye Inner"), (0, 0, "Left Eye"), (0, 0, "Left Eye Outer"),
    (0, 0, "Right Eye Inner"), (0, 0, "Right Eye"), (0, 0, "Right Eye Outer"),
    (0, 0, "Left Ear"), (0, 0, "Right Ear"), (0, 0, "Mouth Left"), (0, 0, "Mouth Right"),
    (27, 18, "Left Shoulder"), (13, 18, "Right Shoulder"), (35, 22, "Left Elbow"), (5, 22, "Right Elbow"),
    (35, 27, "Left Wrist"), (5, 27, "Right Wrist"), (0, 0, "Left Pinky"), (0, 0, "Right Pinky"),
    (0, 0, "Left Index"), (0, 0, "Right Index"), (0, 0, "Left Thumb"), (0, 0, "Right Thumb"),
    (25, 25, "Left Hip"), (15, 25, "Right Hip"), (30, 30, "Left Knee"), (10, 30, "Right Knee"),
    (0, 0, "Left Ankle"), (0, 0, "Right Ankle"), (0, 0, "Left Heel"), (0, 0, "Right Heel"),
    (30, 35, "Left Foot Index"), (10, 35, "Right Foot Index")
]

joint_positions = initial_joint_positions.copy()

matching = []
faces = []
predict_results = []
top3_indices_list = []
top3_probs_list = []

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

class App:
    def load_video(self):
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = 1
        thread.start()

    def update(self):
        start_time = time.time()

        if self.vid:
            if self.play:
                ret, frame = self.vid.get_frame()
                if ret:
                    faces.clear()
                    predict_results.clear()
                    top3_indices_list.clear()
                    top3_probs_list.clear()
                    self.individuals = {}  # Initialize individuals dictionary for each frame
                    self.pose_id = 0  # Reset pose ID for each frame
                    matching.clear()

                    detects = modelDetection(frame)

                    for r in detects:
                        boxes = r.boxes
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            label = f"face {i+1}"
                            
                            #cv2.rectangle(frame, (x1-20, y1-20), (x2+20, y2+20), (0, 255, 0), 2)
                            #cv2.putText(frame, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            face = frame[y1:y2, x1:x2]
                            faces.append(face)

                            results = modelClassify.predict(face)
                            prediction_idx = results[0].probs.data.cpu().numpy().argmax()
                            prediction = class_dict[prediction_idx]
                            predict_results.append(prediction)

                            top3_indices = results[0].probs.data.cpu().topk(3).indices.numpy()
                            top3_probs = results[0].probs.data.cpu().topk(3).values.numpy()

                            top3_indices_list.append(top3_indices)
                            top3_probs_list.append(top3_probs)

                            self.individuals[i] = {'face_box': (x1, y1, x2, y2), 'pose': None} # Create dictionary

                            print(f"Face {i+1}: {prediction}")

                    self.update_image()

                    # Detect people using model
                    people_detects = modelPose(frame)

                    for r in people_detects:
                        boxes = r.boxes
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            person_region = frame[y1:y2, x1:x2]
                            label = f"pose {i+1}"
                            #cv2.putText(frame, label, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                            # Pose Estimation using MediaPipe for each person detected
                            results = pose.process(cv2.cvtColor(person_region, cv2.COLOR_BGR2RGB))
                            if results.pose_landmarks:
                                #mp.solutions.drawing_utils.draw_landmarks(
                                    #person_region, 
                                    #results.pose_landmarks, 
                                    #mp_pose.POSE_CONNECTIONS,
                                    #mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=5),
                                   # mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=5)
                                #)
                                self.assign_pose_to_individual((x1, y1, x2, y2), results.pose_landmarks) # Matching

                            # Place the processed region back into the frame
                            frame[y1:y2, x1:x2] = person_region

                    self.update_stick()
                    
                    self.update_graph(top3_indices_list, top3_probs_list)
                    self.photo = ImageTk.PhotoImage(
                        image=Image.fromarray(frame).resize((self.vid_width, self.vid_height), Image.NEAREST)
                    )
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)  # play frame
                    self.frame += 1
                    self.update_counter(self.frame)

        self.canvas.after(abs(int((self.delay - (time.time() - start_time)) * 1000)), self.update)

    def assign_pose_to_individual(self, person_box, pose_landmarks):
        self.pose_id += 1  # Increment pose ID for each detected pose
        for individual_id in sorted(self.individuals.keys()):
            individual = self.individuals[individual_id]
            face_box = individual['face_box']
            if self.is_within(face_box, person_box):
                self.individuals[individual_id]['pose'] = pose_landmarks
                id = individual_id + 1
                matching.append((id, self.pose_id, pose_landmarks))
                matching.sort(key=lambda x: x[0])  # Sort matching by ID
                # Print to confirm matching
                print(len(matching))
                print(f"Face {id} matched with Pose {self.pose_id}")
                print("-------------------------")
                break  # Once matched, no need to check other individuals

    def is_within(self, box1, box2):
        return (box1[0] < box2[2] and box1[2] > box2[0] and box1[1] < box2[3] and box1[3] > box2[1])

    def set_frame(self, frame_no):
        if self.vid:
            ret, frame = self.vid.goto_frame(frame_no)
            self.frame = frame_no
            self.update_counter(self.frame)
            if ret:
                self.photo = ImageTk.PhotoImage(
                    image=Image.fromarray(frame).resize((self.vid_width, self.vid_height), Image.NEAREST))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def update_counter(self, frame):
        self.window.Element("slider").Update(value=frame)
        self.window.Element("counter").Update(f"{frame}/{self.frames}")

    def update_image(self):
        for i in range(20):  # Ensure 20 containers are updated
            if i < len(faces):
                face = faces[i]
                img = Image.fromarray(cv2.resize(face, (40, 40)))
                bio = BytesIO()
                img.save(bio, format="PNG")
                bio.seek(0)
                self.window[f"selected_face_{i}"].update(data=bio.read())
                self.window[f"predicted_face_{i}"].update(value=f"{i+1}: {predict_results[i]}")
            else:
                # Update with a black image if there are no faces
                black_img = Image.new("RGB", (40, 40), (0, 0, 0))
                bio = BytesIO()
                black_img.save(bio, format="PNG")
                bio.seek(0)
                self.window[f"selected_face_{i}"].update(data=bio.read())
                self.window[f"predicted_face_{i}"].update(value=f"{i+1}: not detected")

    def update_stick(self):
        for i in range(20):  # Ensure 20 containers are updated
            if i < len(matching):
                match = matching[i]
                self.update_joints(match)
            else:
                # Set joint_positions back to initial_joint_positions for stick men without detected faces
                global joint_positions
                joint_positions = initial_joint_positions.copy()
                stick_man_image = self.create_stick_man_image()
                stick_man_bytes = self.convert_image_to_bytes(stick_man_image)
                self.window[f"srick_man_{i}"].update(data=stick_man_bytes)

    def update_joints(self, match):
        global joint_positions
        id, _, pose_landmarks = match
        if pose_landmarks:
            joint_positions = initial_joint_positions.copy()
            for joint_index, (x, y, name) in enumerate(joint_positions):
                if joint_index < len(pose_landmarks.landmark):
                    landmark = pose_landmarks.landmark[joint_index]
                    joint_positions[joint_index] = (
                        int(landmark.x * 40),
                        int(landmark.y * 40),
                        name
                    )
            self.update_stick_man_image(id)

    def update_stick_man_image(self, index):
        stick_man_image = self.create_stick_man_image()
        stick_man_bytes = self.convert_image_to_bytes(stick_man_image)
        self.window[f"srick_man_{index-1}"].update(data=stick_man_bytes)

    def init_graph(self):
        plt.rcParams.update({'font.size': 8}) 
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.line1, = self.ax.plot([], [], label='Class 1')
        self.line2, = self.ax.plot([], [], label='Class 2')
        self.line3, = self.ax.plot([], [], label='Class 3')
        self.ax.legend()

        self.graph_canvas = FigureCanvasTkAgg(self.fig, self.window['graph_canvas'].TKCanvas)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.x_data = []
        self.y_data1 = []
        self.y_data2 = []
        self.y_data3 = []

    def update_graph(self, top3_indices_list, top3_probs_list):
        # This assumes you want to plot the top probabilities of the first detected face
        if len(top3_indices_list) > 0:
            top3_probs = top3_probs_list[0]

            self.x_data.append(self.frame)
            self.y_data1.append(top3_probs[0])
            self.y_data2.append(top3_probs[1])
            self.y_data3.append(top3_probs[2])

            self.line1.set_data(self.x_data, self.y_data1)
            self.line2.set_data(self.x_data, self.y_data2)
            self.line3.set_data(self.x_data, self.y_data3)

            self.ax.relim()
            self.ax.autoscale_view()
            self.graph_canvas.draw()

    def create_stick_man_image(self):
        # Create a blank image with an alpha channel (transparency)
        image = np.zeros((40, 40, 4), dtype="uint8")

        # Set the color to white (including alpha for transparency)
        color = (255, 255, 255, 255)

        # Draw the head (scale down to fit within 40x40)
        cv2.circle(image, joint_positions[0][:2], 5, color, 1)

        # Calculate the middle point between left shoulder and right shoulder
        middle_shoulder = (joint_positions[11][0] + joint_positions[12][0]) // 2

        # Draw the body
        cv2.line(image, (joint_positions[0][0], joint_positions[0][1] + 5), (middle_shoulder, joint_positions[12][1]), color, 1)
        cv2.line(image, joint_positions[11][:2], joint_positions[23][:2], color, 1)
        cv2.line(image, joint_positions[12][:2], joint_positions[24][:2], color, 1)

        # Draw the shoulders
        cv2.line(image, joint_positions[11][:2], joint_positions[12][:2], color, 1)

        # Draw the arms
        cv2.line(image, joint_positions[11][:2], joint_positions[13][:2], color, 1)
        cv2.line(image, joint_positions[12][:2], joint_positions[14][:2], color, 1)
        cv2.line(image, joint_positions[13][:2], joint_positions[15][:2], color, 1)
        cv2.line(image, joint_positions[14][:2], joint_positions[16][:2], color, 1)

        # Draw the hips
        cv2.line(image, joint_positions[23][:2], joint_positions[24][:2], color, 1)

        # Draw the legs
        cv2.line(image, joint_positions[23][:2], joint_positions[25][:2], color, 1)
        cv2.line(image, joint_positions[24][:2], joint_positions[26][:2], color, 1)
        cv2.line(image, joint_positions[25][:2], joint_positions[31][:2], color, 1)
        cv2.line(image, joint_positions[26][:2], joint_positions[32][:2], color, 1)

        # Draw the selected joints
        selected_indices = [11, 12, 23, 24, 13, 14, 15, 16, 25, 26, 31, 32]
        for index in selected_indices:
            x, y, _ = joint_positions[index]  # Ignoring the third element (name)
            if x != 0 or y != 0:  # Check if the joint is not at (0, 0)
                cv2.circle(image, (x, y), 1, color, -1)

        return image

    def convert_image_to_bytes(self, image):
        is_success, im_buf_arr = cv2.imencode(".png", image)
        byte_im = im_buf_arr.tobytes()
        return byte_im

    def __init__(self):
        self.play = True
        self.delay = 0.023
        self.frame = 1
        self.frames = None
        self.vid = None
        self.photo = None

        stick_man_image = self.create_stick_man_image()
        stick_man_bytes = self.convert_image_to_bytes(stick_man_image)

        menu_def = [['&File', ['&Open', '&Save', '---', 'Properties', 'E&xit']],
                    ['&Edit', ['Paste', ['Special', 'Normal'], 'Undo']],
                    ['&Help', '&About...']]
      
        video_play_column = [
            [sg.Text("Video Directory")], [sg.Input(key="_FILEPATH_"), sg.Button("Browse")],
            [sg.Canvas(size=(500, 300), key="canvas", background_color="black")],
            [sg.Slider(size=(30, 20), range=(0, 200), resolution=100, key="slider", orientation="h",
                       enable_events=True), sg.Text("0", key="counter", size=(10, 1))],
            [sg.Button('Next frame'), sg.Button("Pause", key="Play"), sg.Button('Exit')],
            [sg.Text("Prediction: ", key="prediction")],
            [sg.Column([[sg.Canvas(size=(500, 300), key="graph_canvas")]], size=(500, 300), key="graph_canvas")],
        ]

        empty_container_column = [
            [sg.Text("Detected Faces", font=("Helvetica", 14))],
            *[
                [
                    sg.Image(key=f"selected_face_{i}", size=(40, 40), background_color="black"), 
                    sg.Image(key=f"srick_man_{i}", data=stick_man_bytes, size=(40, 40)), 
                    sg.Text(f"{i+1}: not detected", key=f"predicted_face_{i}", size=(15, 1)),
                    sg.Image(key=f"selected_face_{i+1}", size=(40, 40), background_color="black"), 
                    sg.Image(key=f"srick_man_{i+1}", data=stick_man_bytes, size=(40, 40)), 
                    sg.Text(f"{i+2}: not detected", key=f"predicted_face_{i+1}", size=(15, 1))
                ] 
                for i in range(0, 20, 2)
            ]
        ]

        layout = [
            [
                sg.Menu(menu_def),
                sg.Column(empty_container_column, key="empty_container", element_justification='center'),
                sg.Column(video_play_column, element_justification='center'),
            ]
        ]

        self.window = sg.Window('Emotion', layout).Finalize()
        canvas = self.window.Element("canvas")
        self.canvas = canvas.TKCanvas

        self.init_graph()
        self.load_video()

        while True:
            event, values = self.window.Read(timeout=100)

            if event is None or event == 'Exit':
                break
            if event == "Browse":
                video_path = None
                try:
                    video_path = sg.filedialog.askopenfile().name
                except AttributeError:
                    print("no video selected, doing nothing")

                if video_path:
                    print(video_path)
                    self.vid = MyVideoCapture(video_path)
                    self.vid_width = 500
                    self.vid_height = int(self.vid_width * self.vid.height / self.vid.width)
                    self.frames = int(self.vid.frames)
                    self.window.Element("slider").Update(range=(0, int(self.frames)), value=0)
                    self.window.Element("counter").Update(f"0/{self.frames}")
                    self.canvas.config(width=self.vid_width, height=self.vid_height)
                    self.frame = 0
                    self.delay = 1 / self.vid.fps

                    self.window.Element("_FILEPATH_").Update(video_path)

            if event == "Play":
                self.play = not self.play
                self.window.Element("Play").Update("Pause" if self.play else "Play")

            if event == 'Next frame':
                self.set_frame(self.frame + 1)

            if event == "slider":
                self.set_frame(int(values["slider"]))

            if event == sg.WINDOW_CLOSED:
                break

        self.window.Close()
        sys.exit()

class MyVideoCapture:
    def __init__(self, video_source):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return 0, None

    def goto_frame(self, frame_no):
        if self.vid.isOpened():
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.vid.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return 0, None

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

if __name__ == '__main__':
    App()
