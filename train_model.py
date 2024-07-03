import pandas as pd
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('image_labels_with_clusters.csv')

IMG_SIZE = 128
dataset_dir = 'happiness'

def resize_with_aspect_ratio(image, target_size):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_h, target_size - new_h - pad_h,
        pad_w, target_size - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    return padded_image

images = []
labels = []

for index, row in df.iterrows():
    image_path = row['image_path']
    label = row['cluster']
    
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_with_aspect_ratio(image, IMG_SIZE)
        images.append(image)
        labels.append(label)

images = np.array(images, dtype='float32') / 255.0
labels = to_categorical(labels, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

model.save('emotion_intensity_model.h5')

def predict_emotion_intensity(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_with_aspect_ratio(image, IMG_SIZE)
    image = np.expand_dims(image, axis=0) / 255.0
    
    prediction = model.predict(image)
    cluster = np.argmax(prediction)
    intensity = ["Low", "Medium", "High"]
    
    return intensity[cluster], cluster

test_image_path = 'happiness/img_10796.png'
predicted_intensity, predicted_cluster = predict_emotion_intensity(test_image_path)

actual_row = df[df['image_path'] == test_image_path]
actual_intensity = actual_row['emotion'].values[0]
actual_cluster = actual_row['cluster'].values[0]

image = cv2.imread(test_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = resize_with_aspect_ratio(image, IMG_SIZE)

plt.figure(figsize=(5, 5))
plt.imshow(image_resized)
plt.title(f"Predicted: {predicted_intensity} (Cluster {predicted_cluster})\nActual: {actual_intensity} (Cluster {actual_cluster})")
plt.axis('off')
plt.show()
