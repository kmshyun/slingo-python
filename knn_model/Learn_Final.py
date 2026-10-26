import cv2
import mediapipe as mp
import time
import os
import json
import numpy as np

class_labels = ["MTE", "SUNGHWUAN", "Hello", "A", "B", "C", "D","Del", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "Space", "T", "U", "V", "W", "X", "Y", "Z"]
save_root = "LEARNED"
os.makedirs(save_root, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

for label in class_labels:
    save_path = os.path.join(save_root, f"{label}.json")
    print(f"스페이스바 누르면 '{label}' 제스처를 5초간 녹화합니다.")
    collecting = False
    start_time = None
    frames_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if collecting:
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    base = landmarks[0]
                    relative = landmarks - base # 상대좌표 계산

                    # 손이 앞뒤에 있는것도 고려하여 닮음비 계산
                    norm = np.linalg.norm(relative, axis=1).max()
                    if norm > 0:
                        relative = relative / norm

                    frames_data.append(relative.tolist())

        if collecting:
            elapsed = time.time() - start_time
            cv2.putText(frame, f"Recording {label}: {elapsed:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if elapsed >= 7:
                collecting = False
                print(f"'{label}' 녹화 완료. 총 {len(frames_data)}프레임 저장 중...")

                data_array = np.array(frames_data)
                mean_array = np.mean(data_array, axis=0).tolist()

                to_save = {
                    "data": data_array.tolist(),
                    "mean": mean_array
                }
                with open(save_path, 'w') as f:
                    json.dump(to_save, f, indent=2)

                print(f"저장 완료: {save_path}")
                frames_data.clear()
                break

        cv2.imshow("Gesture Recorder", frame)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key == 32 and not collecting:
            print(f"'{label}' 녹화 시작")
            collecting = True
            start_time = time.time()
            frames_data.clear()

cap.release()
cv2.destroyAllWindows()
