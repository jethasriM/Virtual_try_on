import cv2
import numpy as np
import mediapipe as mp
import requests
from io import BytesIO

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detect_pose(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    return results.pose_landmarks

def load_image_from_url(url):
    response = requests.get(url)
    img = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    return img

def overlay_product_on_user(user_image_path, product_image_path, output_image_path):
    user_img = load_image_from_url(user_image_path)
    product_img = load_image_from_url(product_image_path)

    landmarks = detect_pose(user_img)

    if landmarks:
        shoulder_left = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_right = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        h, w, _ = user_img.shape
        shoulder_left = (int(shoulder_left.x * w), int(shoulder_left.y * h))
        shoulder_right = (int(shoulder_right.x * w), int(shoulder_right.y * h))

        distance = np.linalg.norm(np.array(shoulder_right) - np.array(shoulder_left))
        scale = distance / product_img.shape[1]

        product_img_resized = cv2.resize(product_img, (0, 0), fx=scale, fy=scale)

        x_offset = (shoulder_left[0] + shoulder_right[0]) // 2 - product_img_resized.shape[1] // 2
        y_offset = shoulder_left[1] - product_img_resized.shape[0] // 2

        if product_img_resized.shape[2] == 4:  # Check if the image has an alpha channel
            mask = product_img_resized[:, :, 3]
            product_img_rgb = product_img_resized[:, :, :3]

            x_end = min(x_offset + product_img_rgb.shape[1], user_img.shape[1])
            y_end = min(y_offset + product_img_rgb.shape[0], user_img.shape[0])
            x_start = max(x_offset, 0)
            y_start = max(y_offset, 0)

            for c in range(0, 3):
                user_img[y_start:y_end, x_start:x_end, c] = (
                    (1 - mask[(y_start - y_offset):(y_end - y_offset), (x_start - x_offset):(x_end - x_offset)] / 255.0) * 
                    user_img[y_start:y_end, x_start:x_end, c] +
                    (mask[(y_start - y_offset):(y_end - y_offset), (x_start - x_offset):(x_end - x_offset)] / 255.0) * 
                    product_img_rgb[(y_start - y_offset):(y_end - y_offset), (x_start - x_offset):(x_end - x_offset), c]
                )
        else:
            user_img[y_offset:y_offset + product_img_resized.shape[0], 
                     x_offset:x_offset + product_img_resized.shape[1]] = product_img_resized

    cv2.imwrite(output_image_path, user_img)
    print(f'Virtual try-on result saved to {output_image_path}')


user_image_path = 'https://www.shutterstock.com/image-photo/young-elegant-woman-long-straight-260nw-2056664435.jpg'
product_image_path = 'https://i.pinimg.com/236x/15/78/25/1578258285ee62c7047a45da4c72701b.jpg'
output_image_path = 'virtual_try_on_result.jpg'

overlay_product_on_user(user_image_path, product_image_path, output_image_path)
