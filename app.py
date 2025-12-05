import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import io
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

app = Flask(__name__)


# Load Model + Classes

model = tf.keras.models.load_model("models/coconut_model.keras", compile=False)
print("MODEL LOADED SUCCESSFULLY")

with open("models/coconut_classes.json") as f:
    class_names = json.load(f)


# Preprocess Function

def preprocess(img):
    img = img.resize((300, 300)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)


# Home Route (single definition!)

@app.route("/")
def home():
    # renders templates/home.html
    return render_template("home.html")


# Predict Route

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read()))
    
    arr = preprocess(img)
    pred = model.predict(arr)[0]

    idx = int(np.argmax(pred))
    label = class_names[idx]
    confidence = float(pred[idx])

    return jsonify({
        "prediction": label,
        "class_id": idx,
        "confidence": confidence
    })


# Run Server

if __name__ == "__main__":
    app.run(debug=True, port=5000)
