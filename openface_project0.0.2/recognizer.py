from deepface import DeepFace
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from database import collection
import os

def get_embedding(image_path):
    return DeepFace.represent(img_path=image_path, model_name="Facenet")[0]["embedding"]

def add_user(image_path, name):
    embedding = get_embedding(image_path)
    doc = {
        "name": name,
        "embedding": embedding
    }
    collection.insert_one(doc)
    return {"message": f"{name} added successfully."}

def recognize(image_path):
    input_embed = np.array(get_embedding(image_path)).reshape(1, -1)
    min_dist = float("inf")
    identity = "Unknown"

    for doc in collection.find():
        db_embed = np.array(doc["embedding"]).reshape(1, -1)
        dist = euclidean_distances(input_embed, db_embed)[0][0]

        if dist < min_dist and dist < 10:  # Facenet threshold ~10
            min_dist = dist
            identity = doc["name"]

    if min_dist == float("inf"):
        return {"identity": identity, "distance": None}  # Or you can omit distance field
    else:
        return {"identity": identity, "distance": float(min_dist)}
