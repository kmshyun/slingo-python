import os
import numpy as np
import tensorflow as tf
import cv2

# === 설정 ===
IMG_SIZE = 64
MODEL_PATH = 'gesture_classifier.h5'
TEST_IMAGE = 'testerer.jpg' 
DATA_DIR = 'selected_data' 

class_names = sorted(os.listdir(DATA_DIR))

model = tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(TEST_IMAGE):
    print(f" '{TEST_IMAGE}' File not Found.")
    exit()
else:
  print("file Found")

img = cv2.imread(TEST_IMAGE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img.astype('float32') / 255.0
input_img = np.expand_dims(img, axis=0)


prediction = model.predict(input_img, verbose=0)
predicted_index = np.argmax(prediction[0])
predicted_label = class_names[predicted_index]
confidence = np.max(prediction[0])


print(f"예측 결과: {predicted_label} ({confidence:.2%} 확신)")
