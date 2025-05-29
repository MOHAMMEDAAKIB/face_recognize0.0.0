import cv2
import numpy as np

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def save_face_images(username, files):
    import os
    folder_path = f"saved_images/{username}"
    os.makedirs(folder_path, exist_ok=True)
    paths = []

    for idx, file in enumerate(files):
        img = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
        sharp_img = sharpen_image(img)
        save_path = f"{folder_path}/{idx+1}.jpg"
        cv2.imwrite(save_path, sharp_img)
        paths.append(save_path)
    return paths
