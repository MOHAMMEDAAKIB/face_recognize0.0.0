from mtcnn import MTCNN
import cv2
import numpy as np
import os

detector = MTCNN()
model = cv2.dnn.readNetFromTorch("models/openface.nn4.small2.v1.t7")

def get_face_embedding(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Cannot read image")
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if len(faces) == 0:
        print("[INFO] No face detected")
        return None

    face = max(faces, key=lambda x: x["confidence"])
    x, y, w, h = face["box"]
    face_crop = rgb[y:y+h, x:x+w]

    if face_crop.size == 0:
        return None

    face_blob = cv2.dnn.blobFromImage(face_crop, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    model.setInput(face_blob)
    vec = model.forward()
    return vec.flatten()
