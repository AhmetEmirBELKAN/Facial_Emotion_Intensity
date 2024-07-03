import pandas as pd
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('emotion_intensity_model.h5')

IMG_SIZE = 128
dataset_dir = 'happiness'
csv_path = 'image_labels_with_clusters.csv'

df = pd.read_csv(csv_path)

def resize_with_aspect_ratio(image, target_size):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    #paddingamk
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

def predict_emotion_intensity(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_with_aspect_ratio(image, IMG_SIZE)
    image = np.expand_dims(image, axis=0) / 255.0
    
    prediction = model.predict(image)
    cluster = np.argmax(prediction)
    intensity = ["Low", "Medium", "High"]
    
    return intensity[cluster], cluster

for index, row in df.iterrows():
    image_path = row['image_path']
    predicted_intensity, predicted_cluster = predict_emotion_intensity(image_path)

    actual_intensity = row['emotion']
    actual_cluster = row['cluster']

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = resize_with_aspect_ratio(image, IMG_SIZE)

    plt.figure(figsize=(5, 5))
    plt.imshow(image_resized)
    plt.title(f"Predicted: {predicted_intensity} (Cluster {predicted_cluster})\nActual: {actual_intensity} (Cluster {actual_cluster})")
    plt.axis('off')
    plt.show()

    if cv2.waitKey(0) & 0xFF == ord('q'):
        continue

cv2.destroyAllWindows()
