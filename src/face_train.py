import os
import numpy as np
from PIL import Image
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            print(label, path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)


            pile_image = Image.open(path).convert("L")
            size = (550,550)
            final_image = pile_image.resize(size)


            image_array = np.array(final_image, "uint8")
            print(f"Image: {file}, Array Shape: {image_array.shape}")
            # print(image_array)

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            print(f"Image: {file}, Detected faces: {len(faces)}")
            for(x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                # roi = cv2.resize(roi, (100, 100))
                print(f"ROI Shape: {roi.shape}, Label: {id_}")
                x_train.append(roi)
                y_labels.append(id_)

# print(y_label)
# print(x_train)
                
with open("labels.pickle", 'wb')as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")

# if not x_train or not y_labels:
#     print("Error: Empty training data.")
# else:
#     # Train the recognizer
#     training_successful = recognizer.train(x_train, np.array(y_labels))

#     # Check if the training was successful
#     if training_successful:
#         print("Training successful.")
#         # Save the trained recognizer to a file
#         recognizer.save("trainner.yml")
#     else:
#         print("Error: Training failed.")

