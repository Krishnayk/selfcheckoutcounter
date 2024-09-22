from ultralytics import YOLO  ##imported YOLO from ultralytics to use our model to test our video adn detect objects
import cv2                    ## to use the video or webcam as a input and show it with various addons to us, a library
import time                   ##used to use time in the project

# Load the custom-trained YOLO model
model = YOLO('best_model.pt') ## custom trained model(best_model.pt) loaded in variable named model

# Initialize variables / and initial default value setted before the video starts
total_item_count = 0
pen = 0
paper = 0
last_detection_time = 0  # Track the time of the last detection
cooldown_time = 10  # Cooldown time in seconds
detected_items = {}  # Dictionary to track item counts
prices = {
    "house_pen": 10,  # Example item names and their prices
    "house_paper": 5
    # Add more items and prices as needed
}#### remember the classname defined in the model and in the code you write must be same

# --------------------XXXXXXXXXXXXXXX--------------
# Function to process YOLO detection, count items, and print class
def yolo_and_count(frame):
    global total_item_count, last_detection_time, detected_items, pen, paper ## global for using it throughout the code
    # Get the current time
    current_time = time.time()

    # Perform prediction on the current frame
    results = model(frame)

    # Check if any objects were detected in the frame
    for result in results:
        boxes = result.boxes  # Bounding boxes for the detected objects

        if len(boxes) > 0:  # If any items are detected
            # Check if the cooldown time has passed since the last detection
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
                        print(f"Detected: {class_name}")  # Print the detected item

                last_detection_time = current_time  # Update the last detection time

    # Save the frame with detection results (for visualization)
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


# Open video file (replace with "0" for laptop camera)
cap = cv2.VideoCapture("test3.mp4")   ## VideoCaputre is used to use the camera or video as an input

# Check if the video or camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

# Loop to continuously get frames from the video
while True:
    # Capture frame-by-frame
    ret, frame_basic = cap.read()    ##ret: Returns True if the frame was successfully captured, False otherwise.
                                     ##frame_basic: The actual frame/image captured from the video, which you can process or display.

    # If a frame was successfully captured
    if not ret:  # If `ret` is False, it means the video has ended
        print("End of video detected. Exiting...")
        break

    # Run YOLO detection and count items in the frame
    frame = yolo_and_count(frame_basic)

    # Display the frame with YOLO predictions
    cv2.imshow('Item Counter', frame)    ## 'Item Counter': The title of the window.
                                                  ##frame: The image data to be displayed, typically containing detected items and annotations.

    # Press 'q' on the keyboard to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video/camera and close all windows
cap.release()                                    ##closes the window
cv2.destroyAllWindows()                          # Close all OpenCV windows

# Print the final bill after the session ends
display_bill()
