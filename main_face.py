from ultralytics import YOLO  ##imported YOLO from ultralytics to use our model to test our video adn detect objects
import cv2                    ## to use the video or webcam as a input and show it with various addons to us, a library
import time                   ##used to use time in the project


# Load the custom-trained YOLO model
model = YOLO('best_model.pt')## custom trained model(best_model.pt) loaded in variable named model

# Load OpenCV's pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables    and initial default value setted before the video starts
total_item_count = 0
pen = 0
paper = 0
last_detection_time = 0  # Track the time of the last detection
cooldown_time = 10  # Cooldown time in seconds
face_detected = False  # Flag to indicate if a face is detected
detected_items = {}  # Dictionary to track item counts
prices = {
    "house_pen": 10,  # Example item names and their prices
    "house_paper": 5
#### remember the classname defined in the model and in the code you write must be same
}


# Function to process YOLO detection, count items, and print class
def yolo_and_count(frame):
    global total_item_count, last_detection_time, detected_items, pen, paper   ## global for using it throughout the code
    current_time = time.time()

    # Perform prediction on the current frame
    results = model(frame)

    # Check if any objects were detected in the frame
    for result in results:
        boxes = result.boxes  # Bounding boxes for the detected objects
        if len(boxes) > 0:  # If any items are detected
            if current_time - last_detection_time > cooldown_time:
                for i in range(len(boxes)):
                    class_id = int(result.boxes.cls[i])  # Class ID
                    class_name = result.names[class_id]  # Class name

                    # If the class_name is recognized, count the item
                    if class_name in prices:
                        if class_name == "house_pen":
                            pen += 1  # Increment the pen count
                        if class_name == "house_paper":
                            paper += 1  # Increment the paper count

                        detected_items[class_name] = detected_items.get(class_name, 0) + 1  # Track item count
                        print(f"Detected: {class_name}")

                last_detection_time = current_time  # Update the last detection time

    frame_with_detections = result.plot()  # Draw bounding boxes and classes on the frame
    return frame_with_detections


# Function to display the final bill after exiting
def display_bill():
    print("\n--- Final Bill ---")
    total_price = 0
    print(f"house_Pen x{pen}")
    print(f"house_paper x{paper}")

    for item, count in detected_items.items():
        price = prices[item] * count
        total_price += price
        print(f"{item}: {count} x {prices[item]} = {price}")

    print(f"\nTotal Price: {total_price}")


# Open the video capture (0 for laptop camera)
cap = cv2.VideoCapture(0)             ## VideoCaputre is used to use the camera or video as an input

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame_basic = cap.read()     ##ret: Returns True if the frame was successfully captured, False otherwise.
                                      ##frame_basic: The actual frame/image captured from the video, which you can process or display.

    # If a frame was successfully captured
    if ret:
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame_basic, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                ##gray: This is the picture we're looking at, changed to black and white to make it easier to find faces.
                ##scaleFactor=1.1: This setting makes the picture a bit bigger, which helps find faces that might be different sizes.
                ##minNeighbors=5: For something to be recognized as a face, it needs to have at least 5 other nearby rectangles that also say "face." This helps avoid mistakes.
                ##minSize=(30, 30): This sets the smallest size for a face to be found. If a face is too tiny, it won’t be counted.


        # If faces are detected, start object detection
        if len(faces) > 0:
            face_detected = True  # Set the face detection flag to True

        # If a face has been detected, perform YOLO object detection
        if face_detected:
            frame = yolo_and_count(frame_basic)

            # Display the frame with YOLO predictions
            cv2.imshow('Item Counter', frame)
                     ## 'Item Counter': The title of the window.
                     ##frame: The image data to be displayed, typically containing detected items and annotations.

        else:
            # Just show the video stream if no face is detected
            cv2.imshow('Item Counter', frame_basic)

    # Press 'Esc' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == 0x1B:  ##0x1B is the hex code for the Esc key
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()  # Close all OpenCV windows

# Print the final bill after the session ends
display_bill()
