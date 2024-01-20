import dlib
import cv2
import numpy as np
import pickle
import os

# Load the face detector
detector = dlib.get_frontal_face_detector()

# Path to your image folder
image_folder_path = "images"

# Initialize an empty dictionary to store the dataset
dataset = {}

# Iterate through subdirectories (assuming each subdirectory is a label)
for label in os.listdir(image_folder_path):
    label_path = os.path.join(image_folder_path, label)

    # Check if it's a directory (to avoid files in the root folder)
    if os.path.isdir(label_path):
        # List all files in the subdirectory (assuming they are images)
        image_paths = [os.path.join(label_path, file) for file in os.listdir(label_path)]

        # Add the label and corresponding image paths to the dataset
        dataset[label] = image_paths

# Initialize variables to store feature vectors and corresponding labels
X_train = []
y_labels = []

# Loop through the dataset
for label, image_paths in dataset.items():
    for image_path in image_paths:
        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Detect faces using dlib
        faces = detector(img)

        # Loop through detected faces
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            # Ensure the region to be resized is not empty and within the image boundaries
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and w > 0 and h > 0 and (y + h) <= img.shape[0] and (x + w) <= img.shape[1]:
                # Resize the face region to a consistent size (e.g., 100x100)
                roi_gray_resized = cv2.resize(img[y:y+h, x:x+w], (100, 100))

                # Flatten the resized face region as a simple feature
                roi_gray_flattened = roi_gray_resized.flatten()

                # Append the feature vector and label to the training set
                X_train.append(roi_gray_flattened)
                y_labels.append(label)

# Convert labels to integer type
label_dict = {label: i for i, label in enumerate(np.unique(y_labels))}
y_labels_numeric = np.array([label_dict[label] for label in y_labels])

# Convert X_train and y_labels to NumPy arrays
X_train = np.array(X_train, dtype=np.float32)

# Create an SVM model and train it
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(X_train, cv2.ml.ROW_SAMPLE, y_labels_numeric)
svm.save("svm_model.xml")