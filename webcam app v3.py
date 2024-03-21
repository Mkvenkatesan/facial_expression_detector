import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('emotion_detection_model_augmented.h5')

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# List of emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def predict_emotion(frame):
    # Resize frame to match model input shape
    resized_frame = cv2.resize(frame, (640, 480))
    # Convert frame to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    # Expand dimensions to match model input
    gray = np.expand_dims(gray, axis=-1)
    # Normalize pixel values
    gray = gray / 255.0
    # Predict emotion
    predictions = model.predict(np.array([gray]))
    # Get predicted emotion label
    emotion_label = EMOTIONS[np.argmax(predictions[0])]
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

        # Detect faces
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            emotion_label = predict_emotion(face_roi)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
