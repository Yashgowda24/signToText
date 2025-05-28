import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import pyttsx3

# Load model and label map
model = load_model("models/model.h5")
label_map = {0: "hello", 1: "sorry", 2: "thank_you"}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Initialize hands with max_num_hands=2
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# In the main loop:
sequence = []
engine = pyttsx3.init()
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Initialize frame_landmarks for two hands
        frame_landmarks = [None, None]
        
        # Process each detected hand
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            if handedness == 'Left' and frame_landmarks[0] is None:
                frame_landmarks[0] = landmarks
            elif handedness == 'Right' and frame_landmarks[1] is None:
                frame_landmarks[1] = landmarks
            
            # Draw hand landmarks for visualization
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Fill missing hands with zeros
        if frame_landmarks[0] is None:
            frame_landmarks[0] = [0.0] * 63
        if frame_landmarks[1] is None:
            frame_landmarks[1] = [0.0] * 63
        
        # Combine landmarks from both hands
        combined_landmarks = frame_landmarks[0] + frame_landmarks[1]
        sequence.append(combined_landmarks)

        if len(sequence) == 40:
            sequence_np = np.expand_dims(sequence, axis=0)
            prediction = model.predict(sequence_np)
            pred_label = label_map[np.argmax(prediction)]

            print(pred_label)
            engine.say(pred_label)
            engine.runAndWait()
            sequence = []
    else:
        sequence = []

    cv2.imshow("Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break