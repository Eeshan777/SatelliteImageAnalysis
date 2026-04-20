import tensorflow as tf
import numpy as np
import os, cv2, uuid

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vgg16_model.keras")

try:
    classifier = tf.keras.models.load_model(MODEL_PATH, compile=False)
except:
    classifier = None

def analyze(path):
    if classifier is None: return None

    img_cv = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    classes = ["Annual Crop", "Forest", "Herbaceous Veg", "Highway", "Industrial", 
               "Pasture", "Permanent Crop", "Residential", "River", "Sea/Lake"]

    # 10-Class Feature Map Configuration (Name, RGB Color, HSV Lower, HSV Upper)
    feature_map = [
        {"name": "Annual Crop",    "c": [255, 255, 0],   "l": [20, 40, 40],  "u": [35, 255, 255]},
        {"name": "Forest",         "c": [0, 100, 0],     "l": [35, 40, 20],  "u": [85, 255, 150]},
        {"name": "Herbaceous Veg", "c": [173, 255, 47],  "l": [25, 30, 50],  "u": [45, 255, 255]},
        {"name": "Highway",        "c": [128, 128, 128], "l": [0, 0, 50],    "u": [180, 50, 150]},
        {"name": "Industrial",     "c": [255, 0, 0],     "l": [0, 0, 40],    "u": [180, 20, 100]},
        {"name": "Pasture",        "c": [255, 215, 0],   "l": [20, 20, 100], "u": [40, 150, 255]},
        {"name": "Permanent Crop", "c": [34, 139, 34],   "l": [30, 50, 20],  "u": [70, 255, 100]},
        {"name": "Residential",    "c": [255, 105, 180], "l": [0, 0, 150],   "u": [180, 40, 255]},
        {"name": "River",          "c": [0, 255, 255],   "l": [85, 40, 50],  "u": [110, 255, 255]},
        {"name": "Sea/Lake",       "c": [0, 0, 255],     "l": [100, 50, 20], "u": [140, 255, 255]}
    ]

    # --- 1. PATCH ANALYSIS (4x4) ---
    h, w, _ = img_rgb.shape
    ph, pw = h // 4, w // 4 
    all_patch_preds = []
    patch_details = []

    for row in range(4):
        for col in range(4):
            patch = img_rgb[row*ph:(row+1)*ph, col*pw:(col+1)*pw]
            p_input = cv2.resize(patch, (224, 224)) / 255.0
            p_batch = np.expand_dims(p_input, 0).astype('float32')
            p_preds = classifier.predict(p_batch, verbose=0)[0]
            
            # Forest/Sea Bias Correction
            hsv_p = cv2.cvtColor(cv2.resize(patch, (50, 50)), cv2.COLOR_RGB2HSV)
            green_px = np.sum(cv2.inRange(hsv_p, np.array([35, 40, 20]), np.array([85, 255, 255])) > 0)
            if classes[np.argmax(p_preds)] == "Sea/Lake" and green_px > 375:
                p_preds[classes.index("Sea/Lake")] = 0.0
                p_preds[classes.index("Forest")] = 0.9

            all_patch_preds.append(p_preds)
            patch_details.append((f"[{row+1},{col+1}]", classes[np.argmax(p_preds)], f"{np.max(p_preds)*100:.1f}%"))

    # --- 2. GLOBAL STATS ---
    combined = np.mean(all_patch_preds, axis=0)
    model_report = [("GLOBAL", classes[np.argmax(combined)].upper(), f"{np.max(combined)*100:.1f}%")]
    model_report.extend(patch_details)

    # --- 3. SEGMENTATION OVERLAY ---
    overlay = cv2.resize(img_rgb, (800, 800))
    hsv = cv2.cvtColor(cv2.bilateralFilter(overlay, 9, 75, 75), cv2.COLOR_RGB2HSV)
    feature_report = []

    for f in feature_map:
        mask = cv2.inRange(hsv, np.array(f['l']), np.array(f['u']))
        perc = (np.sum(mask > 0) / 640000) * 100
        if perc > 0.1:
            overlay[mask > 0] = cv2.addWeighted(overlay, 0.5, np.full_like(overlay, f['c']), 0.5, 0)[mask > 0]
            # Return Hex Color for UI Legend
            hex_color = '#%02x%02x%02x' % tuple(f['c'])
            feature_report.append((f['name'], f"{perc:.1f}%", hex_color))

    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.abspath(os.path.join(out_dir, f"{uuid.uuid4().hex}.png"))
    cv2.imwrite(full_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {"label": classes[np.argmax(combined)], "model_data": model_report, "feature_data": feature_report, "seg_path": full_path}