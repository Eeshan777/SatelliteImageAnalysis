import tensorflow as tf
import numpy as np
import os, cv2, uuid

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    classifier = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "vgg16_model.keras"), compile=False)
except:
    classifier = None

def analyze(path):
    if classifier is None: return None

    img_cv = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # 1. PREPROCESSING (224x224 for VGG16)
    img_vgg = cv2.resize(img_rgb, (224, 224)) / 255.0
    img_batch = np.expand_dims(img_vgg, 0).astype('float32')

    # 2. NEURAL PREDICTION
    raw_preds = classifier.predict(img_batch, verbose=0)[0]
    classes = ["Annual Crop", "Forest", "Herbaceous Veg", "Highway", "Industrial", 
               "Pasture", "Permanent Crop", "Residential", "River", "Sea/Lake"]

    # --- SPECTRAL OVERRIDE FOR SEA/LAKE BIAS ---
    hsv_small = cv2.cvtColor(cv2.resize(img_rgb, (100, 100)), cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv_small, np.array([30, 25, 5]), np.array([90, 255, 255]))
    
    # If the image is more than 15% green, it's not a Sea/Lake
    if classes[np.argmax(raw_preds)] == "Sea/Lake" and np.sum(green_mask > 0) > 1500:
        raw_preds[classes.index("Sea/Lake")] = 0.0
        primary_label = classes[np.argmax(raw_preds)]
    else:
        primary_label = classes[np.argmax(raw_preds)]

    norm_preds = (raw_preds / np.sum(raw_preds)) * 100
    model_report = [f"{classes[i]}: {norm_preds[i]:.2f}%" for i in norm_preds.argsort()[::-1]]

    # 3. SEGMENTATION (Mapped to VGG16 Classes)
    overlay = cv2.resize(img_rgb, (800, 800))
    refined = cv2.bilateralFilter(overlay, 9, 75, 75)
    hsv = cv2.cvtColor(refined, cv2.COLOR_RGB2HSV)
    
    # Precise color mapping for the 10 VGG classes
    feature_map = [
        {"name": "Forest", "c": [0, 255, 0], "l": [35, 40, 5], "u": [90, 255, 150]},
        {"name": "Sea/Lake", "c": [0, 100, 255], "l": [95, 50, 2], "u": [145, 255, 255]},
        {"name": "Annual Crop", "c": [255, 255, 0], "l": [15, 40, 40], "u": [34, 255, 255]},
        {"name": "Residential", "c": [255, 0, 255], "l": [0, 0, 100], "u": [180, 50, 255]},
        {"name": "River", "c": [0, 255, 255], "l": [85, 30, 10], "u": [100, 255, 180]}
    ]

    pixel_counts = {cls: 0 for cls in classes}
    pixel_counts["Unclassified"] = 0

    for f in feature_map:
        mask = cv2.inRange(hsv, np.array(f['l']), np.array(f['u']))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, cnts, -1, 255, -1)
        
        count = np.sum(mask > 0)
        pixel_counts[f['name']] = count
        
        if count > 0:
            color_layer = np.full_like(overlay, f['c'])
            overlay[mask > 0] = cv2.addWeighted(overlay, 0.5, color_layer, 0.5, 0)[mask > 0]

    # Calculate Unclassified
    total_px = 800 * 800
    classified_sum = sum(pixel_counts.values())
    pixel_counts["Unclassified"] = max(0, total_px - classified_sum)

    feature_report = [f"{cls}: {(pixel_counts[cls]/total_px)*100:.2f}%" for cls in classes if pixel_counts[cls] > 0]
    feature_report.append(f"Unclassified: {(pixel_counts['Unclassified']/total_px)*100:.2f}%")

    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    seg_path = os.path.join(out_dir, f"sync_{uuid.uuid4().hex[:8]}.png")
    cv2.imwrite(seg_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {"label": primary_label, "model_data": model_report, "feature_data": feature_report, "seg_path": seg_path}