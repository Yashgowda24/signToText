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
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize camera and TTS
cap = cv2.VideoCapture(0)
sequence = []
engine = pyttsx3.init()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Use first hand
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:  # 21 landmarks * 3 coords
            sequence.append(landmarks)

        # Draw hand landmarks for visualization
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(sequence) == 41:
            sequence_np = np.expand_dims(sequence, axis=0)
            prediction = model.predict(sequence_np)
            pred_label = label_map[np.argmax(prediction)]

            print(pred_label)
            engine.say(pred_label)
            engine.runAndWait()

            sequence = []  # Reset sequence

    else:
        sequence = []  # Reset sequence if no hands

    cv2.imshow("Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
