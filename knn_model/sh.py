import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time

# 기준 이마 폭 (1 SH = 639px)
BASELINE_SH_PIXELS = 639.0

# 제스처 데이터 불러오기
gesture_db = {}
learned_dir = "LEARNED"

for filename in os.listdir(learned_dir):
    if filename.endswith(".json"):
        label = filename.replace(".json", "")
        with open(os.path.join(learned_dir, filename), 'r') as f:
            data = json.load(f)
            frames = np.array(data["data"])
            gesture_db[label] = frames

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# 제스처 비교 함수
def compare_to_all(current, dataset):
    diffs = dataset - current
    dists = np.mean(diffs ** 2, axis=(1, 2))
    return np.min(dists)

# 변수 초기화
recognized_text = ""
last_recognition_time = 0
recognition_interval = 1.0
recognition_enabled = False

# 카메라 초기화
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
    hand_landmarks_list = []

    # 손 랜드마크 수집
    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # 왼손이면 좌우 반전
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
                hand_landmarks_list.append((hand_landmarks, landmarks))

    # 제스처 비교
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

        # SUNGHWAN 제스처: 인식 활성화 + 이마 측정
        if best_label == "SUNGHWAN":
            recognition_enabled = True
            if len(hand_landmarks_list) == 2:
                p1 = hand_landmarks_list[0][1][8]
                p2 = hand_landmarks_list[1][1][8]
                dist_px = np.linalg.norm((p1[:2] - p2[:2]) * np.array([w, h]))
                sh_value = dist_px / BASELINE_SH_PIXELS
                print(f"[이마 폭 측정] {dist_px:.2f} px → {sh_value:.2f} SH")

        # 다른 제스처는 SUNGHWAN 이후에만 인식
        elif recognition_enabled and best_label:
            if best_label == "Space":
                recognized_text += " "
            elif best_label == "Del":
                recognized_text = recognized_text[:-1]
            else:
                recognized_text += best_label

    # UI 구성
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

    if recognition_enabled:
        cv2.putText(canvas, "Recognition: ON", (w + 10, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 2)
    else:
        cv2.putText(canvas, "Recognition: OFF", (w + 10, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Sign Recognition", canvas)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()