import torch
import torch.nn as nn
import numpy as np
import os, cv2, uuid
from torchvision import transforms, models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- MATCHED ARCHITECTURES ---

def get_vgg16_skeleton():
    model = models.vgg16(weights=None)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(256, 10),
    )
    return model

class SatelliteAutoencoder(nn.Module):
    def __init__(self):
        super(SatelliteAutoencoder, self).__init__()
        # Encoder: 3 -> 32 -> 64
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Decoder: Structurally aligned to fix Missing Key (decoder.4) 
        # and Size Mismatch (192 vs 128)
        self.decoder = nn.Sequential(
            # decoder.0 & .1
            nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            # decoder.2 & .3: Matches shape [192, 64, 3, 3] 
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            # decoder.4 & .5: The missing output layer
            nn.Conv2d(64, 3, 3, padding=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- LOADERS ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pt_model(path, model_type="vgg"):
    if not os.path.exists(path): return None
    try:
        model = get_vgg16_skeleton() if model_type == "vgg" else SatelliteAutoencoder()
        model = model.to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        return model
    except Exception as e:
        print(f"Load Error ({model_type}): {e}")
        return None

classifier = load_pt_model(os.path.join(BASE_DIR, "models", "vgg16_model.pth"), "vgg")
autoencoder = load_pt_model(os.path.join(BASE_DIR, "models", "autoencoder_model.pth"), "ae")

# --- ANALYSIS ENGINE ---

def analyze(path):
    if classifier is None: return None
    img_cv = cv2.imread(path)
    if img_cv is None: return None
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    classes = ["Annual Crop", "Forest", "Herbaceous Veg", "Highway", "Industrial", 
               "Pasture", "Permanent Crop", "Residential", "River", "Sea/Lake"]

    # Full 10-Class Feature Map for masking
    feature_map = [
        {"name": "Annual Crop",    "c": [255, 255, 0],   "l": [20, 40, 40],   "u": [35, 255, 255]},
        {"name": "Forest",         "c": [0, 100, 0],     "l": [35, 50, 20],   "u": [85, 255, 150]},
        {"name": "Herbaceous Veg", "c": [173, 255, 47],  "l": [25, 30, 50],   "u": [45, 150, 255]},
        {"name": "Highway",        "c": [128, 128, 128], "l": [0, 0, 100],    "u": [180, 25, 220]},
        {"name": "Industrial",     "c": [150, 0, 0],     "l": [0, 0, 40],     "u": [180, 45, 120]},
        {"name": "Pasture",        "c": [255, 215, 0],   "l": [20, 100, 100], "u": [40, 255, 255]},
        {"name": "Permanent Crop", "c": [34, 139, 34],   "l": [40, 40, 20],   "u": [70, 255, 100]},
        {"name": "Residential",    "c": [255, 105, 180], "l": [0, 0, 150],    "u": [180, 50, 255]},
        {"name": "River",          "c": [0, 191, 255],   "l": [90, 50, 50],   "u": [110, 255, 200]},
        {"name": "Sea/Lake",       "c": [0, 0, 255],     "l": [110, 50, 20],  "u": [140, 255, 255]}
    ]

    to_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    h, w, _ = img_rgb.shape
    ph, pw = h // 5, w // 5 
    all_patch_preds, patch_details = [], []

    with torch.no_grad():
        for row in range(5):
            for col in range(5):
                patch = img_rgb[row*ph:(row+1)*ph, col*pw:(col+1)*pw]
                tensor = to_tensor(patch).unsqueeze(0).to(device)
                if autoencoder: tensor = autoencoder(tensor)
                
                output = classifier(tensor)
                p_preds = torch.softmax(output, dim=1).cpu().numpy()[0]
                
                # --- HEURISTIC BIAS CORRECTION ---
                current_label = classes[np.argmax(p_preds)]
                gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                # Industrial has complex textures (std_dev > 50). Highways are smooth.
                std_dev = np.std(gray)
                hsv_p = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
                avg_sat = np.mean(hsv_p[:, :, 1])

                if current_label == "Industrial" and std_dev < 45 and avg_sat < 30:
                    p_preds[classes.index("Industrial")] *= 0.05
                    p_preds[classes.index("Highway")] = 0.90

                all_patch_preds.append(p_preds)
                top_idx = np.argmax(p_preds)
                patch_details.append((f"[{row+1},{col+1}]", classes[top_idx], f"{p_preds[top_idx]*100:.1f}%"))

    # Results
    combined = np.mean(all_patch_preds, axis=0)
    final_idx = np.argmax(combined)
    
    # Visualization & Pixel-based Coverage Percentage
    overlay = cv2.resize(img_rgb, (800, 800))
    hsv_full = cv2.cvtColor(cv2.bilateralFilter(overlay, 7, 50, 50), cv2.COLOR_RGB2HSV)
    feature_report = []
    total_px = 800 * 800

    for f in feature_map:
        mask = cv2.inRange(hsv_full, np.array(f['l']), np.array(f['u']))
        coverage = (np.sum(mask > 0) / total_px) * 100
        
        if coverage > 0.5:
            overlay[mask > 0] = cv2.addWeighted(overlay, 0.7, np.full_like(overlay, f['c']), 0.3, 0)[mask > 0]
            feature_report.append((f['name'], f"{coverage:.1f}%", '#%02x%02x%02x' % tuple(f['c'])))

    out_path = os.path.join(BASE_DIR, "outputs", f"{uuid.uuid4().hex}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {
        "label": classes[final_idx], 
        "model_data": [("GLOBAL", classes[final_idx].upper(), f"{combined[final_idx]*100:.1f}%")] + patch_details, 
        "feature_data": feature_report, 
        "seg_path": out_path
    }