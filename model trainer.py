import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Function to capture webcam data and labels
def capture_data(label, num_samples):
    cap = cv2.VideoCapture(0)
    data = []
    
    for _ in range(num_samples):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        data.append([gray, label])
        
        cv2.imshow('Capturing Data...', gray)
        cv2.waitKey(100)
        
    cap.release()
    cv2.destroyAllWindows()
    
    return data

# Collect data for each emotion label
num_samples_per_label = 100  # Adjust as needed
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

all_data = []
for i, emotion in enumerate(emotions):
    print(f"Capturing data for {emotion}...")
    data = capture_data(i, num_samples_per_label)
    all_data.extend(data)

# Preprocess data
X = np.array([entry[0] for entry in all_data])
y = np.array([entry[1] for entry in all_data])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(emotions), activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save('emotion_detection_model.h5')


# Convert the Keras model to TensorFlow's protobuf format (.pb)
tf.saved_model.save(model, 'emotion_detection_model_pb')
