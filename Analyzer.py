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

    # --- 1. REFINED FEATURE MAP (Optimized HSV Ranges) ---
    feature_map = [
        {"name": "Annual Crop",    "c": [255, 255, 0],   "l": [18, 40, 40],  "u": [32, 255, 255]},
        {"name": "Forest",         "c": [0, 100, 0],     "l": [35, 50, 20],  "u": [88, 255, 150]},
        {"name": "Herbaceous Veg", "c": [173, 255, 47],  "l": [25, 35, 50],  "u": [45, 255, 255]},
        {"name": "Highway",        "c": [128, 128, 128], "l": [0, 0, 40],    "u": [180, 45, 140]},
        {"name": "Industrial",     "c": [255, 0, 0],     "l": [0, 0, 30],    "u": [180, 25, 90]},
        {"name": "Pasture",        "c": [255, 215, 0],   "l": [15, 20, 80],  "u": [42, 160, 255]},
        {"name": "Permanent Crop", "c": [34, 139, 34],   "l": [32, 60, 15],  "u": [75, 255, 110]},
        {"name": "Residential",    "c": [255, 105, 180], "l": [0, 0, 140],   "u": [180, 50, 255]},
        {"name": "River",          "c": [0, 255, 255],   "l": [90, 45, 40],  "u": [108, 255, 255]},
        {"name": "Sea/Lake",       "c": [0, 0, 255],     "l": [105, 60, 10], "u": [145, 255, 255]}
    ]

    # --- 2. 5x5 PATCH ANALYSIS (25 PARTS) ---
    h, w, _ = img_rgb.shape
    ph, pw = h // 5, w // 5 
    all_patch_preds = []
    patch_details = []

    for row in range(5):
        for col in range(5):
            patch = img_rgb[row*ph:(row+1)*ph, col*pw:(col+1)*pw]
            p_input = cv2.resize(patch, (224, 224)) / 255.0
            p_batch = np.expand_dims(p_input, 0).astype('float32')
            p_preds = classifier.predict(p_batch, verbose=0)[0]
            
            # STRENGTHENED SPECTRAL GATE
            hsv_p = cv2.cvtColor(cv2.resize(patch, (60, 60)), cv2.COLOR_RGB2HSV)
            green_px = np.sum(cv2.inRange(hsv_p, np.array([35, 50, 20]), np.array([88, 255, 150])) > 0)
            
            # Logic: If high green density, it cannot be Sea/Lake
            if classes[np.argmax(p_preds)] == "Sea/Lake" and green_px > 400: # ~11% threshold
                p_preds[classes.index("Sea/Lake")] = 0.0
                p_preds[classes.index("Forest")] = 0.95

            all_patch_preds.append(p_preds)
            patch_details.append((f"[{row+1},{col+1}]", classes[np.argmax(p_preds)], f"{np.max(p_preds)*100:.1f}%"))

    # --- 3. NORMALIZED GLOBAL STATS ---
    combined = np.mean(all_patch_preds, axis=0)
    model_report = [("GLOBAL", classes[np.argmax(combined)].upper(), f"{np.max(combined)*100:.1f}%")]
    model_report.extend(patch_details)

    # --- 4. SEGMENTATION & 100% TALLY ---
    overlay = cv2.resize(img_rgb, (800, 800))
    hsv = cv2.cvtColor(cv2.bilateralFilter(overlay, 7, 50, 50), cv2.COLOR_RGB2HSV)
    
    raw_counts = []
    for f in feature_map:
        mask = cv2.inRange(hsv, np.array(f['l']), np.array(f['u']))
        cnt = np.sum(mask > 0)
        raw_counts.append(cnt)
        if cnt > 0:
            overlay[mask > 0] = cv2.addWeighted(overlay, 0.6, np.full_like(overlay, f['c']), 0.4, 0)[mask > 0]

    # Normalization Logic to ensure sum == 100
    total_detected = sum(raw_counts)
    if total_detected == 0: total_detected = 1
    
    feature_report = []
    for i, f in enumerate(feature_map):
        percentage = (raw_counts[i] / total_detected) * 100
        if percentage > 0.01: # Filter out negligible noise
            hex_color = '#%02x%02x%02x' % tuple(f['c'])
            feature_report.append((f['name'], f"{percentage:.1f}%", hex_color))

    # Save Output
    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.abspath(os.path.join(out_dir, f"{uuid.uuid4().hex}.png"))
    cv2.imwrite(full_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {"label": classes[np.argmax(combined)], "model_data": model_report, "feature_data": feature_report, "seg_path": full_path}