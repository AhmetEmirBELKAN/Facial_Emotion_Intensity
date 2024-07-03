import time
import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

silhouette = np.array([
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
])

lipsUpperOuter = np.array([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291])
lipsLowerOuter = np.array([146, 91, 181, 84, 17, 314, 405, 321, 375, 291])
lipsUpperInner = np.array([78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308])
lipsLowerInner = np.array([78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308])

rightEyeUpper0 = np.array([246, 161, 160, 159, 158, 157, 173])
rightEyeLower0 = np.array([33, 7, 163, 144, 145, 153, 154, 155, 133])
rightEyeUpper1 = np.array([247, 30, 29, 27, 28, 56, 190])
rightEyeLower1 = np.array([130, 25, 110, 24, 23, 22, 26, 112, 243])
rightEyeUpper2 = np.array([113, 225, 224, 223, 222, 221, 189])
rightEyeLower2 = np.array([226, 31, 228, 229, 230, 231, 232, 233, 244])
rightEyeLower3 = np.array([143, 111, 117, 118, 119, 120, 121, 128, 245])
rightEyeUpper3 = np.array([116, 123, 124, 125, 126, 127, 129, 130, 131])

rightEyebrowUpper = np.array([156, 70, 63, 105, 66, 107, 55, 193])
rightEyebrowLower = np.array([35, 124, 46, 53, 52, 65])

rightEyeIris = np.array([473, 474, 475, 476, 477])

leftEyeUpper0 = np.array([466, 388, 387, 386, 385, 384, 398])
leftEyeLower0 = np.array([263, 249, 390, 373, 374, 380, 381, 382, 362])
leftEyeUpper1 = np.array([467, 260, 259, 257, 258, 286, 414])
leftEyeLower1 = np.array([359, 255, 339, 254, 253, 252, 256, 341, 463])
leftEyeUpper2 = np.array([342, 445, 444, 443, 442, 441, 413])
leftEyeLower2 = np.array([446, 261, 448, 449, 450, 451, 452, 453, 464])
leftEyeLower3 = np.array([372, 340, 346, 347, 348, 349, 350, 357, 465])

leftEyebrowUpper = np.array([383, 300, 293, 334, 296, 336, 285, 417])
leftEyebrowLower = np.array([265, 353, 276, 283, 282, 295])

leftEyeIris = np.array([468, 469, 470, 471, 472])

midwayBetweenEyes = np.array([168])

noseTip = np.array([1])
noseBottom = np.array([2])
noseRightCorner = np.array([98])
noseLeftCorner = np.array([327])

rightCheek = np.array([205])
leftCheek = np.array([425])

# L2 norm 
def calculate_distances(landmarks, indices):
    return np.linalg.norm(landmarks[indices[:-1]] - landmarks[indices[1:]], axis=1)

def extract_features(landmarks):
    upper_lip_outer_distances = calculate_distances(landmarks, lipsUpperOuter)
    lower_lip_outer_distances = calculate_distances(landmarks, lipsLowerOuter)
    upper_lip_inner_distances = calculate_distances(landmarks, lipsUpperInner)
    lower_lip_inner_distances = calculate_distances(landmarks, lipsLowerInner)
    
    right_eye_upper0_distances = calculate_distances(landmarks, rightEyeUpper0)
    right_eye_lower0_distances = calculate_distances(landmarks, rightEyeLower0)
    left_eye_upper0_distances = calculate_distances(landmarks, leftEyeUpper0)
    left_eye_lower0_distances = calculate_distances(landmarks, leftEyeLower0)
    
    right_eyebrow_upper_distances = calculate_distances(landmarks, rightEyebrowUpper)
    left_eyebrow_upper_distances = calculate_distances(landmarks, leftEyebrowUpper)

    features = np.concatenate([
        upper_lip_outer_distances, lower_lip_outer_distances,
        upper_lip_inner_distances, lower_lip_inner_distances,
        right_eye_upper0_distances, right_eye_lower0_distances,
        left_eye_upper0_distances, left_eye_lower0_distances,
        right_eyebrow_upper_distances, left_eyebrow_upper_distances
    ])
    return features

def draw_landmarks(image, landmarks):
    for idx, landmark in enumerate(landmarks):
        x, y = int(landmark[0] * image.shape[1]), int(landmark[1] * image.shape[0])
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    return image

dataset_dir = 'happiness'
features_list = []

for image_file in os.listdir(dataset_dir):
    image_path = os.path.join(dataset_dir, image_file)
    image = cv2.imread(image_path)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print(image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
            features = extract_features(landmarks)
            features_list.append(features)
            
            # image_with_landmarks = draw_landmarks(image.copy(), landmarks)
            # cv2.imshow('Image with Landmarks', image_with_landmarks)
            
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     continue

features_array = np.array(features_list)

df = pd.DataFrame(features_array)
df.to_csv('features.csv', index=False)

cv2.destroyAllWindows()
