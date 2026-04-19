import tensorflow as tf
import numpy as np
import os, cv2, uuid

# Suppress TensorFlow logging for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the VGG16 Model once globally
try:
    classifier = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "models", "vgg16_model.keras"), 
        compile=False
    )
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

def analyze(path):
    if classifier is None:
        return {"label": "ERROR", "model_data": ["Model not found"], "feature_data": [], "seg_path": ""}

    # --- 1. IMAGE PRE-PROCESSING ---
    img_cv = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    # Resize for the VGG16 input (assumed 512x512 based on previous context)
    img_batch = np.expand_dims(cv2.resize(img_rgb, (512, 512)) / 255.0, 0).astype('float32')

    # --- 2. NEURAL DETECTION (The "AI" Report) ---
    pred = classifier.predict(img_batch, verbose=0)[0]
    classes = [
        "Annual Crop", "Forest", "Herbaceous Veg", "Highway", "Industrial", 
        "Pasture", "Permanent Crop", "Residential", "River", "Sea/Lake"
    ]
    
    # Get top 3 predictions for the Model Report
    top_indices = pred.argsort()[-3:][::-1]
    model_report = [f"{classes[idx]}: {pred[idx]*100:.2f}%" for idx in top_indices]
    primary_label = classes[np.argmax(pred)]

    # --- 3. SPECTRAL SEGMENTATION (The "Feature" Report) ---
    overlay = cv2.resize(img_rgb, (800, 800))
    # Apply Gaussian Blur to reduce noise in high-res satellite tiles
    hsv = cv2.cvtColor(cv2.GaussianBlur(overlay, (5,5), 0), cv2.COLOR_RGB2HSV)
    
    # Define color thresholds (Recalibrated based on your reference images)
    # AnnualCrop_2 (Dull greens) vs Forest_4 (Deep dark greens)
    features = [
        {"name": "Arboreal (Forest)", "color": [0, 100, 0], "l": [35, 50, 10], "u": [85, 255, 110]},
        {"name": "Cultivation (Crop)", "color": [173, 255, 47], "l": [20, 25, 40], "u": [45, 255, 255]},
        {"name": "Hydrology (Water)", "color": [0, 160, 255], "l": [100, 160, 20], "u": [135, 255, 255]},
        {"name": "Infrastructure", "color": [180, 180, 180], "l": [0, 0, 150], "u": [180, 50, 255]}
    ]

    feature_report = []
    kernel = np.ones((5,5), np.uint8)

    for f in features:
        mask = cv2.inRange(hsv, np.array(f['l']), np.array(f['u']))
        
        # --- LOGIC SYNC: BIAS CORRECTION ---
        # If AI says Forest, but color mask thinks it's Water (likely shadows), erase the water mask
        if primary_label == "Forest" and "Hydrology" in f['name']:
            mask = cv2.erode(mask, kernel, iterations=3)
        
        # Clean up the mask (removing small stray pixels)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate percentage of total image
        perc = (np.sum(mask > 0) / (800 * 800)) * 100
        feature_report.append(f"{f['name']}: {perc:.2f}%")
        
        # Apply color overlay to the segmented image
        if perc > 0.3:
            color_layer = np.full_like(overlay, f['color'])
            overlay[mask > 0] = cv2.addWeighted(overlay, 0.65, color_layer, 0.35, 0)[mask > 0]

    # --- 4. EXPORT OUTPUT ---
    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    seg_path = os.path.join(out_dir, f"analysis_{unique_id}.png")
    
    # Save as BGR for OpenCV
    cv2.imwrite(seg_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {
        "label": primary_label,
        "model_data": model_report,
        "feature_data": feature_report,
        "seg_path": seg_path
    }