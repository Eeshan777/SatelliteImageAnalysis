import tensorflow as tf
import numpy as np
from PIL import Image
import json
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vgg16_model.h5")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs") 
OUTPUT_FILE = "report.json"

model = tf.keras.models.load_model(MODEL_PATH)

classes = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

def detect_domain(label):
    satellite = ["Forest", "River", "SeaLake", "Highway", "Residential", "Industrial"]
    return "Satellite Image" if label in satellite else "General Image"

def analyze(path):
    img = Image.open(path).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, 0)
    
    pred = model.predict(img)
    index = np.argmax(pred)
    label = classes[index]
    confidence = float(np.max(pred))
    domain = detect_domain(label)
    
    metadata = {
        "file": path,
        "label": label,
        "domain": domain,
        "confidence": confidence,
        "timestamp": str(datetime.now())
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    history = []
    if os.path.exists(full_output_path):
        with open(full_output_path, "r") as f:
            try:
                history = json.load(f)
                if not isinstance(history, list):
                    history = [history]
            except json.JSONDecodeError:
                history = []

    history.append(metadata)

    with open(full_output_path, "w") as f:
        json.dump(history, f, indent=4)
    
    return metadata