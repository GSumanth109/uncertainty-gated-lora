import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image
import numpy as np
import os

# 1. SETUP
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running Baseline Evaluation on: {device.upper()}")

# 2. LOAD STANDARD MODEL (The "Static" Baseline)
print("Loading standard ResNet-50 (ImageNet Pre-trained)...")
model_id = "microsoft/resnet-50"
processor = AutoImageProcessor.from_pretrained(model_id)
model = ResNetForImageClassification.from_pretrained(model_id).to(device)
model.eval()

def get_entropy(logits):
    """Calculates Shannon Entropy: How confused is the model?"""
    probs = F.softmax(logits, dim=-1)
    # Entropy = -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy.item()

def evaluate_folder(folder_path, description):
    """Runs the model on a folder of images and tracks 'Confusion'."""
    print(f"\n--- Testing Domain: {description.upper()} ---")
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path} (Skipping)")
        return

    entropies = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
    
    if len(image_files) == 0:
        print("No images found.")
        return

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        image = Image.open(img_path).convert("RGB")
        
        inputs = processor(image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            
        # We don't care about the *class* (Cat vs Dog).
        # We care about the *CONFIDENCE* (Entropy).
        ent = get_entropy(logits)
        entropies.append(ent)
        
    avg_entropy = np.mean(entropies)
    print(f"Processed {len(image_files)} images.")
    print(f"Average Model Entropy (Confusion): {avg_entropy:.4f}")
    
    if avg_entropy < 1.0:
        print("✅ Verdict: Model is Confident.")
    else:
        print("❌ Verdict: Model is CONFUSED (Domain Shift Detected).")

# 3. EXECUTION
# You/Friends will put sample images in these folders
evaluate_folder("data/test_sunny", "Sunny (Baseline)")
evaluate_folder("data/test_rain", "Rain (OOD)")
evaluate_folder("data/test_night", "Night (OOD)")
