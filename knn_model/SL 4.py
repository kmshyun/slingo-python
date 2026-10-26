import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

model = tf.keras.models.load_model('model.keras')
class_names = sorted(os.listdir('root'))  # root 폴더의 클래스 순서와 동일하게 불러옴

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_image)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = image.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            margin = 30
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            hand_img = image[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            hand_img = cv2.resize(hand_img, (300, 300))
            input_tensor = np.expand_dims(hand_img / 255.0, axis=0)

            predictions = model.predict(input_tensor, verbose=0)
            class_index = np.argmax(predictions[0])
            class_label = class_names[class_index]
            confidence = tf.nn.softmax(predictions[0])[class_index].numpy()

            cv2.putText(image, f"{class_label} ({confidence:.2f})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Recognition", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


'''
광주시 오룡동 광과고
명사모와 밥먹으며 놀아요
씨엔씨엘 들이키며 으라차
이인호쌤 백을타고 짜잔짠
 
시간이 가면 잊지않아요
명사모와한 비밀이야기
행운을 빌어주세요

새로운 채점법

김명용 김명용
팔자는 델타라

김명용 김명용
엘오는 비자야

김명용 김명용
ㄹㅇㅋㅋ만 치세요

새로운 채점법

김명용 김명용
팔자는 델타라

김명용 김명용
엘오는 비자야

김명용 김
광곽을 떠나야지

김명용 김명용
팔자는 델타라

김명용 김명용
엘오는 비자야

김명용 김
광곽을 떠나야지
'''

'''
세포내 디네인 키네신
방추사를 주므르며 놀아요
씨디케이 두알먹고 으라차
세포분열 에너지가 짜잔짠
 
몇년이가도 계속늘어요
비알씨에이 원발암세포
분열좀 그만합시다

새로운 딸세포

성상체 성상체
후기에 키네신

성상체 성상체
아이고 방추사

성상체 성상체
세포분열좀 합시다

새로운 딸세포

성상체 성상체
후기에 키네신

성상체 성상체
아이고 방추사

성상체 성
세포자 살해야지

성상체 성상체
후기에 키네신

성상체 성상체
아이고 방추사

성상체 성
세포자 살해야지
'''


'''


[Spoken]
나효찬!
위대한 국민시대!
나효찬!
위대한 국민시대!
(하! 하! 하!)
나효찬!
나효찬!
(하! 하! 하!)
나효찬!

[Verse 1]
5번!
(하!)
나효찬!
(하!)
5번!
(하!)
나효찬!
(하!)
나효찬 5번 5번 나효찬 5번

[Chorus]
당당한 대통령 나효찬 5번
당당한 대통령 나효찬 5번
서민의 대통령 나효찬 5번

[Verse 2]
학생 복지로 꿈과 희망이
5번!
넘치는 광주과고 나효찬!
(예!)
광주과고 위기 극복할 방법
5번!
지학이 답이다. (기호 오번 나효찬)
(오!)


'''