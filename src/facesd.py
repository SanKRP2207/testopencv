# import cv2
# import dlib
# import pickle
# import numpy as np

# # Load the face detector
# detector = dlib.get_frontal_face_detector()

# # Load the SVM model
# svm = cv2.ml.SVM_load("svm_model.xml")

# labels = {"person_name": 1}  # Adjust based on your labels
# with open("labels.pickle", 'rb') as f:
#     og_labels = pickle.load(f)
#     labels = {v: k for k, v in og_labels.items()}

# # Load the pre-trained eye detector from OpenCV
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces using dlib
#     faces = detector(gray)

#     for face in faces:
#         x, y, w, h = face.left(), face.top(), face.width(), face.height()

#         # Resize the face region to a consistent size (e.g., 100x100)
#         roi_gray_resized = cv2.resize(gray[y:y+h, x:x+w], (100, 100))

#         # Flatten the resized face region as a simple feature
#         roi_gray_flattened = roi_gray_resized.flatten()

#         # Convert to np.float32 if needed
#         roi_gray_flattened = np.float32(roi_gray_flattened)

#         # Ensure that roi_gray_flattened is 1D
#         roi_gray_flattened = roi_gray_flattened.flatten()

#         # Predict using SVM
#         _, result = svm.predict(roi_gray_flattened.reshape(1, -1))

#         # Display the recognized name
#         predicted_label = int(result[0, 0])
#         name = labels.get(predicted_label, "Unknown")

#         font = cv2.FONT_HERSHEY_SIMPLEX
#         color = (255, 255, 255)
#         stroke = 2
#         cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

#         # Draw rectangle around the face
#         color = (255, 0, 0)  # BGR color
#         stroke = 2
#         end_cord_x = x + w
#         end_cord_y = y + h
#         cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

#         # Detect eyes within the face region
#         eyes = eye_cascade.detectMultiScale(roi_gray_resized)
        
#         # Draw rectangles around the detected eyes
#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import dlib
import pickle
import numpy as np

# Load the face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file

# Load the SVM model
svm = cv2.ml.SVM_load("svm_model.xml")

labels = {"person_name": 1}  # Adjust based on your labels

og_labels = None
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# Load the pre-trained eye detector from OpenCV
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using dlib
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Use dlib to find facial landmarks (including eye points)
        shape = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        # Determine if the face is turned to the right or left
        eye_left = landmarks[36:42]  # Points for the left eye
        eye_right = landmarks[42:48]  # Points for the right eye

        left_eye_center = np.mean(eye_left, axis=0).astype(int)
        right_eye_center = np.mean(eye_right, axis=0).astype(int)

        # Calculate the angle between the eyes
        angle = np.degrees(np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0]))

        # Resize the face region to a consistent size (e.g., 100x100)
        roi_gray_resized = cv2.resize(gray[y:y+h, x:x+w], (100, 100))

        # Flatten the resized face region as a simple feature
        roi_gray_flattened = roi_gray_resized.flatten()

        # Convert to np.float32 if needed
        roi_gray_flattened = np.float32(roi_gray_flattened)

        # Ensure that roi_gray_flattened is 1D
        roi_gray_flattened = roi_gray_flattened.flatten()

        # Predict using SVM
        _, result = svm.predict(roi_gray_flattened.reshape(1, -1))

        # Display the recognized name
        predicted_label = int(result[0, 0])
        name = labels.get(predicted_label, "Unknown")

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        # Draw rectangle around the face
        color = (255, 0, 0)  # BGR color
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # Draw rectangle around the eyes based on face orientation
        eye_color = (0, 255, 0)  # BGR color for eyes
        eye_stroke = 2

        if angle < -15:  # Face turned to the left
            for (ex, ey, ew, eh) in eye_left:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), eye_color, eye_stroke)
        elif angle > 15:  # Face turned to the right
            for (ex, ey, ew, eh) in eye_right:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), eye_color, eye_stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
