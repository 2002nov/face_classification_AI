import cv2

# Path to the video file
video_path = '/home/rvp/Work_Student2024/Bow/videotest03.mp4'
# Path to the Haar Cascade file for face detection
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read until the video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret:
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Video', frame)
        
        # Press 'q' on the keyboard to exit the video
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# When everything is done, release the video capture object
cap.release()

# Close all the frames
cv2.destroyAllWindows()


