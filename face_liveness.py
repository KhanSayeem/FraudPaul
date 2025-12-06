import cv2
import time

def run_liveness_detection():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    time.sleep(1)  # allow camera to warm up

    if not cap.isOpened():
        print("Camera failed to open.")
        return False

    movement_threshold = 20
    positions = []
    liveness_confirmed = False

    print("Move your head left/right to pass liveness...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            positions.append(x)
            if len(positions) > 30:
                positions.pop(0)

            movement_range = max(positions) - min(positions)

            if movement_range > movement_threshold:
                liveness_confirmed = True
                cv2.putText(frame, "LIVENESS PASSED", (10,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

                cv2.imshow("Liveness Detection", frame)
                cv2.waitKey(1000)   # show for 1 second
                break
            else:
                cv2.putText(frame, "MOVE HEAD...", (10,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        else:
            cv2.putText(frame, "NO FACE DETECTED", (10,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Liveness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return liveness_confirmed


# Run it directly for testing
if __name__ == "__main__":
    result = run_liveness_detection()
    print("Liveness Passed?", result)
