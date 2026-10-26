import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time

gesture_db = {}
learned_dir = "LEARNED"

for filename in os.listdir(learned_dir):
    if filename.endswith(".json"):
        label = filename.replace(".json", "")
        with open(os.path.join(learned_dir, filename), 'r') as f:
            data = json.load(f)
            frames = np.array(data["data"])
            gesture_db[label] = frames

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

def compare_to_all(current, dataset):
    diffs = dataset - current
    dists = np.mean(diffs ** 2, axis=(1, 2))
    return np.min(dists)
  
recognized_text = ""
last_recognition_time = 0
recognition_interval = 1.0

cap = cv2.VideoCapture(0)
cv2.namedWindow("Sign Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sign Recognition", 1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape
    canvas = np.ones((h, w + 400, 3), dtype=np.uint8) * 255
    current_landmarks = None

    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            if result.multi_handedness:
                handedness = result.multi_handedness[i].classification[0].label
                if handedness == "Left":
                    landmarks[:, 0] = 1.0 - landmarks[:, 0]
                    
            base = landmarks[0]
            relative = landmarks - base
            norm = np.linalg.norm(relative, axis=1).max()
            if norm > 0:
                relative = relative / norm
                current_landmarks = relative

    current_time = time.time()
    if current_landmarks is not None and (current_time - last_recognition_time > recognition_interval):
        last_recognition_time = current_time
        min_dist = float('inf')
        best_label = None
        for label, dataset in gesture_db.items():
            dist = compare_to_all(current_landmarks, dataset)
            if dist < min_dist:
                min_dist = dist
                best_label = label

        if best_label:
            if best_label == "Space":
                recognized_text += " "
            elif best_label == "Del":
                recognized_text = recognized_text[:-1]
            else:
                recognized_text += best_label

    canvas[:, :w] = frame

    cv2.rectangle(canvas, (w, 0), (w + 400, h), (240, 240, 240), -1)
    cv2.putText(canvas, "Recognized Text:", (w + 10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    y = 120
    lines = [recognized_text[i:i + 15] for i in range(0, len(recognized_text), 15)]
    for line in lines:
        cv2.putText(canvas, line, (w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        y += 60

    cv2.imshow("Sign Recognition", canvas)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()