from flask import Flask, request, jsonify
from recognize import get_face_embedding
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from database import collection
from register import register_blueprint

import numpy as np

app = Flask(__name__)
app.register_blueprint(register_blueprint)

@app.route("/verify-face", methods=["POST"])
def verify_face():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    img_path = "images/temp_verify.jpg"
    img.save(img_path)

    emb = get_face_embedding(img_path)
    if emb is None:
        return jsonify({"match": False, "message": "Face not detected"})

    for person in collection.find():
        print(f"Checking person: {person['name']}")
        for idx, stored_emb in enumerate(person["embeddings"]):
            sim = cosine_similarity([emb], [stored_emb])[0][0]
            print(f"  Embedding {idx} similarity: {sim}")
            if sim > 0.9:
                return jsonify({
                    "match": True,
                    "name": person["name"],
                    "nic": person["nic"],
                    "gsdivisionNo": person["gsdivisionNo"],
                    "similarity": float(sim)
                })

    return jsonify({"match": False, "message": "No match found"})

if __name__ == "__main__":
    app.run(debug=True)
