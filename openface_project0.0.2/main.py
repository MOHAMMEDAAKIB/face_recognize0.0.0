from fastapi import FastAPI, UploadFile, File, Form
from face_utils import save_face_images
from database import collection
from deepface import DeepFace
import cv2
import numpy as np

app = FastAPI()

@app.post("/add_user")
async def add_user(name: str = Form(...), files: list[UploadFile] = File(...)):
    image_paths = save_face_images(name, files)
    collection.insert_one({"name": name, "images": image_paths})
    return {"message": f"{name} added with {len(image_paths)} images."}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    input_img = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    users = collection.find()
    best_match = {"identity": "Unknown", "distance": float("inf")}

    for user in users:
        for image_path in user["images"]:
            try:
                result = DeepFace.verify(img1_path=input_img, img2_path=image_path, enforce_detection=False)
                if result["verified"] and result["distance"] < best_match["distance"]:
                    best_match["identity"] = user["name"]
                    best_match["distance"] = result["distance"]
            except Exception as e:
                print("Error:", e)

    return best_match
