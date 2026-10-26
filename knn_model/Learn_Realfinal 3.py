import cv2
import mediapipe as mp
import time
import os
import json
import numpy as np

class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

save_root = "LEARNED"
os.makedirs(save_root, exist_ok=True)
i = 0

video_path = f"0000{i}.avi" 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("영상 파일을 열 수 없습니다.")
    exit()

# 각 라벨마다 영상을 따로 준비했다고 가정하고 반복
for label in class_labels:
    save_path = os.path.join(save_root, f"{label}.json")
    print(f"'{label}' 제스처 영상에서 데이터 수집 중...")

    frames_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"'{label}' 영상 끝.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                base = landmarks[0]
                relative = landmarks - base

                norm = np.linalg.norm(relative, axis=1).max()
                if norm > 0:
                    relative = relative / norm

                frames_data.append(relative.tolist())
                frame_count += 1

        cv2.imshow("Gesture Recorder", frame)
        if cv2.waitKey(1) == 27:  # ESC로 강제 종료
            break

    if frames_data:
        print(f"{label} 수집 완료. 총 {frame_count}프레임 저장 중...")

        data_array = np.array(frames_data)
        mean_array = np.mean(data_array, axis=0).tolist()

        to_save = {
            "data": data_array.tolist(),
            "mean": mean_array
        }
        with open(save_path, 'w') as f:
            json.dump(to_save, f, indent=2)

        print(f"저장 완료: {save_path}")
        i += 1

    cap.release()
    cap = cv2.VideoCapture(video_path)  # 다음 라벨을 위해 영상 재시작

cv2.destroyAllWindows()