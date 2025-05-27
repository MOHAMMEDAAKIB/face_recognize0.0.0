from flask import Blueprint, request, jsonify
from PIL import Image
import os
import numpy as np
from recognize import get_face_embedding
from database import collection

register_blueprint = Blueprint("register", __name__)

@register_blueprint.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    nic = request.form.get("nic")
    gsdivisionNo = request.form.get("gsdivisionNo")
    files = request.files.getlist("images")

    if not name or not nic or not gsdivisionNo or not files:
        return jsonify({"error": "Missing required fields"}), 400

    if len(files) < 3 or len(files) > 5:
        return jsonify({"error": "Please upload between 3 and 5 images"}), 400

    embeddings = []

    for idx, file in enumerate(files):
        img_path = f"images/temp_{name}_{idx}.jpg"
        img = Image.open(file.stream).convert("RGB")
        img.save(img_path)

        emb = get_face_embedding(img_path)
        if emb is None:
            return jsonify({"error": f"Face not detected in image {file.filename}"}), 400

        embeddings.append(emb.tolist())
        os.remove(img_path)  # clean up

    person = {
        "name": name,
        "nic": nic,
        "gsdivisionNo": gsdivisionNo,
        "embeddings": embeddings
    }

    collection.insert_one(person)
    return jsonify({"message": "Person registered successfully!"})
