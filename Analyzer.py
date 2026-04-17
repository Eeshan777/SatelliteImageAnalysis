import tensorflow as tf
import numpy as np
import os
import cv2
import uuid 

# Suppress technical logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure this matches your new .keras model name
MODEL_PATH = os.path.join(BASE_DIR, "models", "vgg16_model.keras")

def analyze(path):
    # 1. Prepare Model
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Load and Preprocess Image
    img_cv = cv2.imread(path)
    if img_cv is None:
        return {"label": "Error: File Not Found", "confidence": 0, "report": [], "seg_path": ""}
        
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
    
    # 3. Neural Prediction
    pred = model.predict(np.expand_dims(img_resized, 0), verbose=0)[0]
    classes = ["Annual Crop", "Forest", "Herbaceous Veg", "Highway", "Industrial", 
               "Pasture", "Permanent Crop", "Residential", "River", "Sea/Lake"]
    
    # Initial guess from the AI
    neural_idx = np.argmax(pred)
    primary_label = classes[neural_idx]
    confidence = float(np.max(pred))

    # 4. Spectral Analysis (Pixel Math)
    # We use a 600x600 canvas for standardizing the stats
    overlay = cv2.resize(img_rgb, (600, 600))
    hsv = cv2.cvtColor(overlay, cv2.COLOR_RGB2HSV)
    
    # Defining Feature Ranges
    # Note: 'Deep Forest' and 'Sea' often overlap; these ranges are tuned to split them.
    features = [
        {"name": "Water Bodies", "color": [0, 168, 255], "l": [100, 60, 40], "u": [130, 255, 255]},
        {"name": "Dense Forest", "color": [39, 174, 96], "l": [35, 45, 30], "u": [90, 255, 255]},
        {"name": "Urban Structures", "color": [149, 165, 166], "l": [0, 0, 120], "u": [180, 40, 255]},
        {"name": "Arid / Soil", "color": [230, 126, 34], "l": [10, 50, 50], "u": [25, 255, 255]}
    ]

    stats_dict = {}
    for feat in features:
        mask = cv2.inRange(hsv, np.array(feat['l']), np.array(feat['u']))
        pixel_count = np.sum(mask > 0)
        percentage = (pixel_count / (600 * 600)) * 100
        stats_dict[feat['name']] = percentage
        
        # Apply visual segmentation overlay
        if percentage > 0.5:
            overlay[mask > 0] = feat['color']

    # --- 5. THE BIAS CORRECTOR (HEURISTIC OVERRIDE) ---
    # This prevents the AI from calling a green area "Sea/Lake"
    
    forest_p = stats_dict["Dense Forest"]
    water_p = stats_dict["Water Bodies"]
    urban_p = stats_dict["Urban Structures"]

    # Rule A: If spectral analysis is CERTAIN it's Forest (e.g. > 50%), trust the math.
    if forest_p > 50 and primary_label != "Forest":
        primary_label = "Forest (Spectral Verification)"
    
    # Rule B: If the model says Sea/Lake but we found 0% water pixels, it's a false positive.
    elif primary_label == "Sea/Lake" and water_p < 2:
        if forest_p > 20:
            primary_label = "Dense Forest"
        elif urban_p > 20:
            primary_label = "Urban / Industrial"
        else:
            primary_label = "Herbaceous Vegetation"

    # Rule C: If the model says Industrial but math sees majority forest
    elif primary_label == "Industrial" and forest_p > 40:
        primary_label = "Forest (Spectral Correction)"

    # 6. Save Segmented Output
    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_filename = f"seg_{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(out_dir, out_filename)
    
    # Convert RGB back to BGR for OpenCV saving
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # Format report for the GUI Treeview
    report = [f"{name}: {val:.2f}%" for name, val in stats_dict.items()]
    
    return {
        "label": primary_label,
        "confidence": confidence,
        "report": report,
        "seg_path": out_path
    }