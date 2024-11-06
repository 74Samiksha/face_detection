import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

def detect_faces(video_path=None, image_path=None, window_width=800, window_height=600):
    # Configure window settings
    window_name = 'Face Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)

    # Calculate the center position for the window
    screen_width, screen_height = 1920, 1080  # Modify according to your screen resolution
    x_pos = (screen_width - window_width) // 2
    y_pos = (screen_height - window_height) // 2
    cv2.moveWindow(window_name, x_pos, y_pos)

    # If an image path is provided, read the image; otherwise, use the video file or webcam
    if image_path:
        # For a static image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detect_and_display_faces(gray, image, window_name)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif video_path:
        # Open a connection to the video file
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detect_and_display_faces(gray, frame, window_name)
            # Press 'q' to exit the video playback early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release the video and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
    else:
        # Open a connection to the webcam
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detect_and_display_faces(gray, frame, window_name)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release the webcam and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

def detect_and_display_faces(gray, frame, window_name):
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Detected {len(faces)} face(s)")
    
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    cv2.imshow(window_name, frame)

# Run the function with a video path for video face detection
# detect_faces(video_path='samples/sample.mp4', window_width=800, window_height=600)  # Adjust width and height as needed
# Or run with detect_faces() for webcam, or 
detect_faces(image_path='samples/pic.jpeg') 
