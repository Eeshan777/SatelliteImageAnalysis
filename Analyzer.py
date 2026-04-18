import tensorflow as tf
import numpy as np
import os
import cv2
import uuid 

# Suppress technical logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Paths
CLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "vgg16_model.keras")
AUTOENCODER_PATH = os.path.join(BASE_DIR, "models", "autoencoder_model.keras")

# --- GLOBAL MODELS ---
# Load once at the start so the analyze function stays fast
print("Initializing 512x512 Neural Engines...")
try:
    # compile=False avoids the 'Adam optimizer' mismatch warnings
    classifier = tf.keras.models.load_model(CLASSIFIER_PATH, compile=False)
    autoencoder = tf.keras.models.load_model(AUTOENCODER_PATH, compile=False)
    print("Models loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load models. {e}")
    classifier = None
    autoencoder = None

def analyze(path):
    if classifier is None or autoencoder is None:
        return {"label": "Model Error: Not Loaded", "confidence": 0, "report": [], "seg_path": "", "recon_path": ""}

    # 1. Load and Preprocess
    img_cv = cv2.imread(path)
    if img_cv is None:
        return {"label": "Error: File Not Found", "confidence": 0, "report": [], "seg_path": "", "recon_path": ""}
        
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # 2. MATCH NEW RESOLUTION (512x512)
    img_resized = cv2.resize(img_rgb, (512, 512))
    img_batch = np.expand_dims(img_resized / 255.0, 0).astype('float32')

    # 3. AUTOENCODER RECONSTRUCTION
    reconstructed_batch = autoencoder.predict(img_batch, verbose=0)
    recon_img = (reconstructed_batch[0] * 255).astype('uint8')
    
    # Save Reconstruction
    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    recon_path = os.path.join(out_dir, f"recon_{uuid.uuid4().hex[:8]}.png")
    cv2.imwrite(recon_path, cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))

    # 4. NEURAL CLASSIFICATION
    # We feed the 'img_batch' (original pixels) or 'reconstructed_batch'
    # Feeding the original resized image usually yields higher accuracy for VGG16
    pred = classifier.predict(img_batch, verbose=0)[0]
    
    classes = ["Annual Crop", "Forest", "Herbaceous Veg", "Highway", "Industrial", 
               "Pasture", "Permanent Crop", "Residential", "River", "Sea/Lake"]
    
    primary_label = classes[np.argmax(pred)]
    confidence = float(np.max(pred))

    # 5. SPECTRAL ANALYSIS (HSV Masking)
    # Using 600x600 for the visual report to keep it sharp
    overlay = cv2.resize(img_rgb, (600, 600))
    refined = cv2.bilateralFilter(overlay, 9, 75, 75)
    hsv = cv2.cvtColor(refined, cv2.COLOR_RGB2HSV)
    
    features = [
        {"name": "Water Bodies", "color": [0, 168, 255], "l": [100, 60, 40], "u": [130, 255, 255]},
        {"name": "Dense Forest", "color": [39, 174, 96], "l": [35, 45, 30], "u": [90, 255, 255]},
        {"name": "Urban Structures", "color": [149, 165, 166], "l": [0, 0, 120], "u": [180, 40, 255]},
        {"name": "Arid / Soil", "color": [230, 126, 34], "l": [10, 50, 50], "u": [25, 255, 255]}
    ]

    stats_dict = {}
    kernel = np.ones((5,5), np.uint8)

    for feat in features:
        mask = cv2.inRange(hsv, np.array(feat['l']), np.array(feat['u']))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        pixel_count = np.sum(mask > 0)
        percentage = (pixel_count / (600 * 600)) * 100
        stats_dict[feat['name']] = percentage
        
        if percentage > 0.5:
            color_layer = np.full_like(overlay, feat['color'])
            mask_bool = mask > 0
            overlay[mask_bool] = cv2.addWeighted(overlay, 0.5, color_layer, 0.5, 0)[mask_bool]

    # 6. BIAS CORRECTION (Manual Override)
    forest_p = stats_dict["Dense Forest"]
    water_p = stats_dict["Water Bodies"]

    if forest_p > 50 and primary_label == "Sea/Lake":
        primary_label = "Forest (Verified)"
    elif primary_label == "Sea/Lake" and water_p < 2:
        primary_label = "Herbaceous Vegetation"

    # 7. Save and Return Output
    seg_path = os.path.join(out_dir, f"seg_{uuid.uuid4().hex[:8]}.png")
    cv2.imwrite(seg_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    report = [f"{name}: {val:.2f}%" for name, val in stats_dict.items()]
    
    return {
        "label": primary_label,
        "confidence": confidence,
        "report": report,
        "seg_path": seg_path,
        "recon_path": recon_path
    }