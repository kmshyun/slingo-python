import cv2  # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore
import os

# 모델과 클래스 이름 로딩
model = tf.keras.models.load_model('gesture_classifier.h5')
IMG_SIZE = 64
class_names = sorted(os.listdir('selected_data'))  # A ~ Z

# 웹캠 켜기
cap = cv2.VideoCapture(0)
print("🖐️ 손을 화면 중앙에 대세요. 'q' 누르면 종료됩니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임 오류")
        break

    h, w, _ = frame.shape

    # 화면 중앙에 ROI (정사각형)
    roi_size = 800
    cx, cy = w // 2, h // 2
    x1, y1 = cx - roi_size // 2, cy - roi_size // 2
    x2, y2 = cx + roi_size // 2, cy + roi_size // 2

    # ROI 잘라서 전처리
    hand_img = frame[y1:y2, x1:x2]
    hand_img_resized = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
    hand_img_norm = hand_img_resized.astype('float32') / 255.0
    input_data = np.expand_dims(hand_img_norm, axis=0)

    # 예측
    prediction = model.predict(input_data, verbose=0)
    index = np.argmax(prediction[0])
    label = class_names[index]
    confidence = np.max(prediction[0])

    # 화면에 표시
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} ({confidence:.0%})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # 출력
    print(f"🔤 인식 결과: {label} ({confidence:.1%})")

    cv2.imshow("Gesture Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()
