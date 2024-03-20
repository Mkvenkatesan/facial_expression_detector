import cv2
import numpy as np

# Load pre-trained face detection and expression recognition models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = cv2.dnn.readNetFromTensorflow("emotion_detection_model.pb")

# List of emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def detect_emotion(face_roi):
    face_roi = cv2.resize(face_roi, (48, 48))
    face_roi = np.expand_dims(face_roi, axis=0)
    face_roi = face_roi / 255.0

    emotion_model.setInput(face_roi)
    emotion_preds = emotion_model.forward()
    emotion_label = EMOTIONS[emotion_preds[0].argmax()]
    
    return emotion_label

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Couldn't open the webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't capture a frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            emotion_label = detect_emotion(face_roi)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
