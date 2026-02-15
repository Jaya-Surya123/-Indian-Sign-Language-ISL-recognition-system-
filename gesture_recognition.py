import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import os

# Load the trained model
model = load_model(r"C:\Users\jayas\OneDrive\Desktop\cnn model\gesture_model.h5")
print(model.input_shape)

# Ensure class_names only contains valid directories
class_names = sorted([
    d for d in os.listdir(r"C:\Users\jayas\OneDrive\Desktop\cnn model\Indian")
    if os.path.isdir(os.path.join(r"C:\Users\jayas\OneDrive\Desktop\cnn model\Indian", d))
])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video capture device.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Initialize variables for combined bounding box
        x_min_combined, y_min_combined = float('inf'), float('inf')
        x_max_combined, y_max_combined = float('-inf'), float('-inf')

        for hand_landmarks in results.multi_hand_landmarks:
            # Get the bounding box of the hand
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Update combined bounding box
            x_min_combined = min(x_min_combined, x_min)
            y_min_combined = min(y_min_combined, y_min)
            x_max_combined = max(x_max_combined, x_max)
            y_max_combined = max(y_max_combined, y_max)

        # Ensure the combined bounding box is within the frame boundaries
        x_min_combined = max(0, x_min_combined)
        y_min_combined = max(0, y_min_combined)
        x_max_combined = min(w, x_max_combined)
        y_max_combined = min(h, y_max_combined)

        # Extract the combined ROI
        roi = frame[y_min_combined:y_max_combined, x_min_combined:x_max_combined]

        # Preprocess the ROI for prediction
        if roi.size > 0:
            resized = cv2.resize(roi, (128, 128))  # Resize to model input size
            normalized = resized.astype("float32") / 255.0  # Normalize pixel values
            reshaped = np.expand_dims(normalized, axis=0)   # Add batch dimension

            # Make predictions
            prediction = model.predict(reshaped)
            class_index = np.argmax(prediction)
            confidence = prediction[0][class_index]

            # Debug: Print prediction details
            print("Prediction:", prediction)
            print("Class index:", class_index)
            print("Confidence:", confidence)

            # Safeguard against invalid class_index and add confidence threshold
            if confidence > 0.5 and class_index < len(class_names):
                gesture_name = class_names[class_index]
            else:
                gesture_name = "Unknown"

            # Draw the combined ROI and prediction on the full frame
            cv2.rectangle(frame, (x_min_combined, y_min_combined), (x_max_combined, y_max_combined), (0, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {gesture_name}", (x_min_combined, y_min_combined - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the full frame with the combined ROI overlay
    cv2.imshow("Real-Time Gesture Recognition", frame)

    # Exit on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()