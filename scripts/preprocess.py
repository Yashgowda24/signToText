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

# Print dataset information
print("\n=== Dataset Information ===")
print("Files found in dataset directory:", os.listdir(DATA_DIR))
print("Looking for videos in:", os.path.abspath(DATA_DIR))
print(f"Found {len(os.listdir(DATA_DIR))} video files\n")

def extract_landmarks(video_path):
    print(f"\nStarting processing: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
        
    sequence = []
    frame_count = 0
    hands_detected = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hands_detected += 1
            frame_landmarks = [None, None]  # [left_hand, right_hand]
            
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                if handedness == 'Left' and frame_landmarks[0] is None:
                    frame_landmarks[0] = landmarks
                elif handedness == 'Right' and frame_landmarks[1] is None:
                    frame_landmarks[1] = landmarks
            
            if frame_landmarks[0] is None:
                frame_landmarks[0] = [0.0] * 63
            if frame_landmarks[1] is None:
                frame_landmarks[1] = [0.0] * 63
            
            combined_landmarks = frame_landmarks[0] + frame_landmarks[1]
            sequence.append(combined_landmarks)
    
    cap.release()
    
    # Print processing summary for this video
    print(f"Finished processing: {os.path.basename(video_path)}")
    print(f"Total frames: {frame_count}")
    print(f"Frames with hands detected: {hands_detected} ({hands_detected/frame_count:.1%})")
    if len(sequence) > 0:
        print(f"Landmark sequence shape: {np.array(sequence).shape}")
    else:
        print("Warning: No hand landmarks detected in this video!")
    
    return np.array(sequence)

# Main processing loop
success_count = 0
for video_file in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, video_file)
    if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
        print(f"Skipping non-video file: {video_file}")
        continue
        
    label = os.path.splitext(video_file)[0]
    data = extract_landmarks(path)
    
    if data is not None and len(data) > 0:
        np.save(os.path.join(OUTPUT_DIR, f"{label}.npy"), data)
        success_count += 1

# Final summary
print("\n=== Processing Complete ===")
print(f"Successfully processed {success_count}/{len(os.listdir(DATA_DIR))} videos")
print(f"Output files created in {OUTPUT_DIR}: {len(os.listdir(OUTPUT_DIR))}")
if success_count > 0:
    print("Preprocessing completed successfully!")
else:
    print("Warning: No valid videos were processed!")