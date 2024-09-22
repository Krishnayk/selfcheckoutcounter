import cv2  # Import the OpenCV library for image and video processing

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the default camera (0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream")  # Print an error message if the camera fails to open
    exit()  # Exit the program if the camera cannot be accessed

while True:  # Start an infinite loop to continuously capture frames
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()

    # Convert the captured frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:  # Loop through each detected face's coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a blue rectangle around each face

    # Display the resulting frame with face detections in a window
    cv2.imshow('Face Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if the user presses the 'q' key

# Release the camera and close all OpenCV windows
cap.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close all OpenCV windows
