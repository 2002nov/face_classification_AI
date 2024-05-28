import sys
import threading
import time
import tkinter as tk
import PIL
from PIL import Image, ImageTk
import cv2
import PySimpleGUI as sg
import os
from PIL import ImageFont
from PIL import ImageDraw
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# import YOLO model
from ultralytics import YOLO

# Load a model
model = YOLO(r"C:\Users\Texxy\Desktop\Emotion\train14-20240513T065634Z-001\train14\weights\best.pt")

class_dict = {0: 'Angry', 1: 'Bored', 2: 'Confused', 3: 'Cool', 4: 'Errrr', 5: 'Funny', 6: 'Happy', 7: 'Normal', 8: 'Proud', 9: 'Sad', 10: 'Scared', 11: 'Shy', 12: 'Sigh', 13: 'Superangry', 14: 'Surprised', 15: 'Suspicious', 16: 'Unhappy', 17: 'Worried', 18: 'sweet', 19: 'tricky'}

class_predict = []

class App:
    """
    TODO: change slider resolution based on vid length
    TODO: make top menu actually do something :P
    """
    def load_video(self):
        """Start video display in a new thread"""
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = 1
        thread.start()

    def update(self):
        """Update the canvas element with the next video frame recursively"""
        start_time = time.time()
        if self.vid:
            if self.play:
                # Get a frame from the video source only if the video is supposed to play
                ret, frame = self.vid.get_frame()
                if ret:
                    # ทำการทำนาย
                    results = model.predict(frame)
                    prediction_idx = results[0].probs.data.argmax().item()
                    prediction = class_dict[prediction_idx]

                    # Get top 3 predictions with their probabilities
                    top3_indices = results[0].probs.data.topk(3).indices.numpy()
                    top3_probs = results[0].probs.data.topk(3).values.numpy()

                    # แสดงผลการทำนายใน terminal
                    print("=====>", prediction)
                    print(self.frame)

                    # อัปเดตองค์ประกอบ text ใน GUI ด้วยผลการทำนาย
                    self.window.Element("prediction").Update(f"Prediction: {prediction}", font=("Calibri", 12))

                    self.photo = PIL.ImageTk.PhotoImage(
                        image=PIL.Image.fromarray(frame).resize((self.vid_width, self.vid_height), Image.NEAREST)
                    )
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                    self.frame += 1
                    self.update_counter(self.frame)

                    # Update the graph
                    self.update_graph(top3_indices, top3_probs)

        self.canvas.after(abs(int((self.delay - (time.time() - start_time)) * 1000)), self.update)

    def set_frame(self, frame_no):
        """Jump to a specific frame"""
        if self.vid:
            ret, frame = self.vid.goto_frame(frame_no)
            self.frame = frame_no
            self.update_counter(self.frame)

            if ret:
                self.photo = PIL.ImageTk.PhotoImage(
                    image=PIL.Image.fromarray(frame).resize((self.vid_width, self.vid_height), Image.NEAREST))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def update_counter(self, frame):
        """Helper function for updating slider and frame counter elements"""
        self.window.Element("slider").Update(value=frame)
        self.window.Element("counter").Update("{}/{}".format(frame, self.frames))

    def init_graph(self):
        """Initialize the matplotlib graph"""
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

    def update_graph(self, top3_indices, top3_probs):
        """Update the graph with new data"""
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

    def __init__(self):
        # ------ App states ------ #
        self.play = True  # Is the video currently playing?
        self.delay = 0.023  # Delay between frames - not sure what it should be, not accurate playback
        self.frame = 1  # Current frame
        self.frames = None  # Number of frames
        # ------ Other vars ------ #
        self.vid = None
        self.photo = None
        self.next = "1"
        self.lx = 0
        self.ly = 0

        folder = r"C:\Users\Texxy\Desktop\Classes"  # get list of PNG files in folder
        png_files = [folder + '/' + f for f in os.listdir(folder) if '.png' in f]
        png_Addfiles = folder + '//////' + '19.png'
        image_size = (40, 40)  # image size
        headers, size = 1, 50  # table cells
        cols, rows = 1, 10  # view cells
        x0, y0 = 0, 0  # Top-left conner

        path = [[(y, x) for x in range(headers)] for y in range(size)]  # refer to image
        print("path:", path)

        def create_image(y, x):
            font = ImageFont.truetype("courbd.ttf", 30)
            im = Image.new('RGBA', image_size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(im, 'RGBA')
            draw.text((image_size[0] // 2, image_size[1] // 2), f'Image ({y}, {x})',
                      font=font, anchor='mm')
            with BytesIO() as output:
                im.save(output, format="PNG")
                im_data = output.getvalue()
            return im_data

        table = [[sg.Image(size=(40, 40), background_color='green', key=(y, x))
                  for x in range(cols)] for y in range(rows)]
        option = {'resolution': 1, 'pad': (0, 0), 'disable_number_display': True,
                  'enable_events': True}

        # ------ Menu Definition ------ #
        menu_def = [['&File', ['&Open', '&Save', '---', 'Properties', 'E&xit']],
                    ['&Edit', ['Paste', ['Special', 'Normal', ], 'Undo'], ],
                    ['&Help', '&About...']]

        # รูปของ 20 class
        file_list_column = [
            [sg.Text("Classes")],
            [sg.Image(r"C:\Users\Texxy\Desktop\Image\0.png"), sg.Text("0 Angry   "),sg.Image(r"C:\Users\Texxy\Desktop\Image\10.png"), sg.Text("10 Scared")],
            [sg.Image(r"C:\Users\Texxy\Desktop\Image\1.png"), sg.Text("1 Bored   "),sg.Image(r"C:\Users\Texxy\Desktop\Image\11.png"), sg.Text("11 Shy")],
            [sg.Image(r"C:\Users\Texxy\Desktop\Image\2.png"), sg.Text("2 Confused"),sg.Image(r"C:\Users\Texxy\Desktop\Image\12.png"), sg.Text("12 Sigh")],
            [sg.Image(r"C:\Users\Texxy\Desktop\Image\3.png"), sg.Text("3 Cool    "),sg.Image(r"C:\Users\Texxy\Desktop\Image\13.png"), sg.Text("13 Superangry")],
            [sg.Image(r"C:\Users\Texxy\Desktop\Image\4.png"), sg.Text("4 Errrr   "),sg.Image(r"C:\Users\Texxy\Desktop\Image\14.png"), sg.Text("14 Surprised")],
            [sg.Image(r"C:\Users\Texxy\Desktop\Image\5.png"), sg.Text("5 Funny   "),sg.Image(r"C:\Users\Texxy\Desktop\Image\15.png"), sg.Text("15 Suspicious")],
            [sg.Image(r"C:\Users\Texxy\Desktop\Image\6.png"), sg.Text("6 Happy   "),sg.Image(r"C:\Users\Texxy\Desktop\Image\16.png"), sg.Text("16 Sweet")],
            [sg.Image(r"C:\Users\Texxy\Desktop\Image\7.png"), sg.Text("7 Normal  "),sg.Image(r"C:\Users\Texxy\Desktop\Image\17.png"), sg.Text("17 Tricky")],
            [sg.Image(r"C:\Users\Texxy\Desktop\Image\8.png"), sg.Text("8 Proud   "),sg.Image(r"C:\Users\Texxy\Desktop\Image\18.png"), sg.Text("18 Unhappy")],
            [sg.Image(r"C:\Users\Texxy\Desktop\Image\9.png"), sg.Text("9 Sad     "),sg.Image(r"C:\Users\Texxy\Desktop\Image\19.png"), sg.Text("19 Worried")],
        ]


        video_play_column = [
            [sg.Text("Video Directory")], [sg.Input(key="_FILEPATH_"), sg.Button("Browse")],
            [sg.Canvas(size=(500, 300), key="canvas", background_color="black")],
            [sg.Slider(size=(30, 20), range=(0, 200), resolution=100, key="slider", orientation="h",
                       enable_events=True), sg.T("0", key="counter", size=(10, 1))],
            [sg.Button('Next frame'), sg.Button("Pause", key="Play"), sg.Button('Exit')],
            [sg.Text("Prediction: ", key="prediction")],
            [sg.Canvas(size=(300, 200), key="graph_canvas")],
        ]

        layout = [
            [
                sg.Menu(menu_def),
                sg.Column(file_list_column),
                sg.Column(video_play_column, element_justification='center'),
            ]
        ]

        self.window = sg.Window('Emotion', layout).Finalize()
        canvas = self.window.Element("canvas")
        self.canvas = canvas.TKCanvas

        # Initialize the graph
        self.init_graph()

        # Start video display thread
        self.load_video()

        while True:  # Main event Loop
            event, values = self.window.Read(timeout=100)

            if event is None or event == 'Exit':
                """Handle exit"""
                break
            if event == "Browse":
                """Browse for files when the Browse button is pressed"""
                # Open a file dialog and get the file path
                video_path = None
                try:
                    video_path = sg.filedialog.askopenfile().name
                except AttributeError:
                    print("no video selected, doing nothing")

                if video_path:
                    print(video_path)
                    # Initialize video
                    self.vid = MyVideoCapture(video_path)
                    # Calculate new video dimensions
                    self.vid_width = 500
                    self.vid_height = int(self.vid_width * self.vid.height / self.vid.width)
                    self.frames = int(self.vid.frames)
                    # Update slider to match amount of frames
                    self.window.Element("slider").Update(range=(0, int(self.frames)), value=0)
                    # Update right side of counter
                    self.window.Element("counter").Update("0/%i" % self.frames)
                    # change canvas size approx to video size
                    self.canvas.config(width=self.vid_width, height=self.vid_height)

                    # Reset frame count
                    self.frame = 0
                    self.delay = 1 / self.vid.fps

                    # Update the video path text field
                    self.window.Element("_FILEPATH_").Update(video_path)

            if event == "Play":
                if self.play:
                    self.play = False
                    self.window.Element("Play").Update("Play")
                else:
                    self.play = True
                    self.window.Element("Play").Update("Pause")

            if event == 'Next frame':
                # Jump forward a frame TODO: let user decide how far to jump
                self.set_frame(self.frame + 1)

            if event == "slider":
                self.set_frame(int(values["slider"]))

            if event == sg.WINDOW_CLOSED:
                break

        self.window.Close()
        sys.exit()


class MyVideoCapture:
    """
    Defines a new video loader with openCV
    Original code from https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
    Modified by me
    """
    def __init__(self, video_source):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        """
        Return the next frame
        """
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return 0, None

    def goto_frame(self, frame_no):
        """
        Go to specific frame
        """
        if self.vid.isOpened():
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)  # Set current frame
            ret, frame = self.vid.read()  # Retrieve frame
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return 0, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


if __name__ == '__main__':
    App()
