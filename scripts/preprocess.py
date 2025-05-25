import cv2
import mediapipe as mp
import os
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = "dataset"
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import os
print("Looking for videos in:", os.path.abspath(DATA_DIR))

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                sequence.append(landmarks)
    cap.release()
    return np.array(sequence)

for video_file in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, video_file)
    label = os.path.splitext(video_file)[0]
    data = extract_landmarks(path)
    np.save(os.path.join(OUTPUT_DIR, f"{label}.npy"), data)
    print(f"Processing video: {video_file}")
