
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow

# Load YOLO
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []

with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f]

# Open the video capture
cap = cv2.VideoCapture('your_video_file.mp4')  # Replace 'your_video_file.mp4' with the path to your video file
 
# Read the input image
input_image_path = 'input_image.png'  # Replace with the path to your input image file
input_image = cv2.imread(input_image_path)

# Ensure the input image dimensions match the YOLO network input size
input_image = cv2.resize(input_image, (416, 416))

# Flag to indicate whether the person is present in any frame
person_present = False

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if there are no more frames

    # Get frame dimensions and prepare the frame for YOLO
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the YOLO network
    net.setInput(blob)

    # Get output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Forward pass to get output from the output layer
    detections = net.forward(output_layer_names)

    # Process YOLO detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'person':
                person_present = True

    # Display the result using cv2_imshow
    cv2_imshow(frame)

    # Break the loop if 'q' key is pressed or person is present
    if cv2.waitKey(30) & 0xFF == ord('q') or person_present:
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()

# Check if the person from the input image is present in the video frames
if person_present:
    print("Person is present in the video.")
else:
    print("Person is not present in the video.")
