import cv2
import os
import numpy as np
import pickle

dataset_path = "face_data"
labels = {}
faces = []
ids = []
current_id = 0

# Load images
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            
            if label not in labels:
                labels[label] = current_id
                current_id += 1
            
            face_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            faces.append(face_img)
            ids.append(labels[label])

faces = np.array(faces)
ids = np.array(ids)

# Train LBPH model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, ids)

# Save model + labels
model.write("face_model.yml")
with open("face_labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("LBPH face training complete!")
