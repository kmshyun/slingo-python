import os
import shutil
import numpy as np
import tensorflow as tf  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore

# === 설정 ===
SRC_DIRS = ['root', 'root2']   # 복수 폴더
DST_DIR = 'selected_data'      # 샘플링 결과 저장
STEP = 15                      # 샘플 간격
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15

# === 1. 샘플링 ===
print("1️⃣ 이미지 샘플링 중...")

if not os.path.exists(DST_DIR):
    os.makedirs(DST_DIR)

for src in SRC_DIRS:
    for gesture_folder in os.listdir(src):
        gesture_path = os.path.join(src, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue

        dst_gesture_path = os.path.join(DST_DIR, gesture_folder)
        os.makedirs(dst_gesture_path, exist_ok=True)

        images = sorted(os.listdir(gesture_path))
        sampled = images[::STEP]

        for img in sampled:
            src_img_path = os.path.join(gesture_path, img)
            dst_img_path = os.path.join(dst_gesture_path, f"{src}_{img}")  # 이름 충돌 방지
            shutil.copyfile(src_img_path, dst_img_path)

print("✅ 샘플링 완료: 총", len(os.listdir(DST_DIR)), "클래스\n")

# === 2. 학습 ===
print("2️⃣ 학습 시작...")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.5,
    horizontal_flip=True,
    zoom_range=0.1
)

train_data = datagen.flow_from_directory(
    DST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    Conv2D(32, (3, 3), activation='relu'), MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'), MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'), Dropout(0.3),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, epochs=EPOCHS, validation_data=val_data)

model.save('gesture_classifier.h5')
print("✅ 모델 저장 완료: gesture_classifier.h5")
