import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import os

df = pd.read_csv('features.csv')
features_array = df.values

kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(features_array)

df['label'] = labels
df.to_csv('features_with_labels.csv', index=False)

plt.scatter(features_array[:, 0], features_array[:, 1], c=labels, cmap='viridis')
plt.title('K-means Clustering of Face Features')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

dataset_dir = 'happiness'

output_data = []

def draw_landmarks(image, landmarks):
    for idx, landmark in enumerate(landmarks):
        x, y = int(landmark[0] * image.shape[1]), int(landmark[1] * image.shape[0])
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    return image

for image_file, label in zip(os.listdir(dataset_dir), labels):
    image_path = os.path.join(dataset_dir, image_file)
    image = cv2.imread(image_path)
    print(image)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
            
            # image_with_landmarks = draw_landmarks(image.copy(), landmarks)
            # label_text = f'Cluster: {label}'
            # cv2.putText(image_with_landmarks, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.imshow('Image with Cluster Label', image_with_landmarks)
            
            emotion = "happiness"
            output_data.append([image_path, emotion, label])
            
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()

output_df = pd.DataFrame(output_data, columns=['image_path', 'emotion', 'cluster'])
output_df.to_csv('image_labels_with_clusters.csv', index=False)
