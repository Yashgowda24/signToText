# import cv2
# import mediapipe as mp
# import numpy as np
# from keras.models import load_model
# import pyttsx3

# # Load model and label map
# model = load_model("models/model.h5")
# label_map = {0: "hello", 1: "sorry", 2: "thank_you"}

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands

# # Initialize hands with max_num_hands=2
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# # In the main loop:
# sequence = []
# engine = pyttsx3.init()
# cap = cv2.VideoCapture(0)
# mp_drawing = mp.solutions.drawing_utils

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         # Initialize frame_landmarks for two hands
#         frame_landmarks = [None, None]
        
#         # Process each detected hand
#         for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
#             handedness = results.multi_handedness[i].classification[0].label
#             landmarks = []
#             for lm in hand_landmarks.landmark:
#                 landmarks.extend([lm.x, lm.y, lm.z])
            
#             if handedness == 'Left' and frame_landmarks[0] is None:
#                 frame_landmarks[0] = landmarks
#             elif handedness == 'Right' and frame_landmarks[1] is None:
#                 frame_landmarks[1] = landmarks
            
#             # Draw hand landmarks for visualization
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
#         # Fill missing hands with zeros
#         if frame_landmarks[0] is None:
#             frame_landmarks[0] = [0.0] * 63
#         if frame_landmarks[1] is None:
#             frame_landmarks[1] = [0.0] * 63
        
#         # Combine landmarks from both hands
#         combined_landmarks = frame_landmarks[0] + frame_landmarks[1]
#         sequence.append(combined_landmarks)

#         if len(sequence) == 40:
#             sequence_np = np.expand_dims(sequence, axis=0)
#             prediction = model.predict(sequence_np)
#             pred_label = label_map[np.argmax(prediction)]

#             print(pred_label)
#             engine.say(pred_label)
#             engine.runAndWait()
#             sequence = []
#     else:
#         sequence = []

#     cv2.imshow("Sign Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# Part 2
# Train with image datasets

# import cv2
# import numpy as np
# from keras.models import load_model
# import pyttsx3
# import json

# # Load model and label map
# model = load_model("models/static_sign_model.h5")
# with open("models/label_map.json", "r") as f:
#     label_map = json.load(f)
# label_map = {int(k): v for k, v in label_map.items()}

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# img_height = 64
# img_width = 64

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Preprocess the frame
#     img = cv2.resize(frame, (img_width, img_height))
#     img_array = np.expand_dims(img, axis=0) / 255.0

#     # Predict
#     predictions = model.predict(img_array)
#     confidence = np.max(predictions)
#     predicted_class = np.argmax(predictions)

#     if confidence > 0.8:
#         label = label_map[predicted_class]
#         cv2.putText(frame, f"Prediction: {label}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         engine.say(label)
#         engine.runAndWait()
#     else:
#         message = "Unrecognized sign."
#         cv2.putText(frame, message, (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#         engine.say(message)
#         engine.runAndWait()

#     # Display the frame
#     cv2.imshow("Static Sign Recognition", frame)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from keras.models import load_model
import pyttsx3
import json
from collections import deque

# Load model and label map
model = load_model("models/static_sign_model.h5")
with open("models/label_map.json", "r") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize webcam
cap = cv2.VideoCapture(0)

img_height = 64
img_width = 64

# For stable prediction
prediction_buffer = deque(maxlen=15)
last_announced_label = None
stable_threshold = 10  # Minimum times same prediction must appear in buffer

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (img_width, img_height))
    img_array = np.expand_dims(img, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array, verbose=0)
    confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)

    label = label_map[predicted_class] if confidence > 0.8 else "Unrecognized"
    prediction_buffer.append(label)

    # Check if current label is stable
    if label != "Unrecognized" and prediction_buffer.count(label) > stable_threshold:
        if label != last_announced_label:
            # Display and speak only when label is new and stable
            cv2.putText(frame, f"Prediction: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            engine.say(label)
            engine.runAndWait()
            last_announced_label = label
        else:
            # Label stable but already announced
            cv2.putText(frame, f"Prediction: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Sign is not stable or unrecognized
        cv2.putText(frame, "Hold your sign steady...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Static Sign Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
