import cv2
import os

# Create directory
dataset_path = "face_data/user"
os.makedirs(dataset_path, exist_ok=True)

# Load Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
count = 0
max_images = 100

print("Starting face capture... Look at the camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200,200))
        
        count += 1
        cv2.imwrite(f"{dataset_path}/img_{count}.jpg", face)
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Images: {count}/{max_images}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Capturing Face Images", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
print("Face data collection complete!")
