import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Allow large Sentinel-1 SAR images

import numpy as np
import matplotlib.pyplot as plt

def preprocess_sar_image(path, size=(224, 224)):
    """Load, resize, normalize SAR .tif image."""
    img = Image.open(path).convert("L")  # Grayscale
    img = img.resize(size)
    arr = np.array(img) / 255.0
    return arr

if __name__ == "__main__":
    sar_folder = "data/raw/sar"
    output_folder = "data/preprocessed/sar"
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(sar_folder):
        print("❌ SAR folder not found:", sar_folder)
        exit()

    files = [f for f in os.listdir(sar_folder) if f.lower().endswith((".tif", ".tiff"))]

    if not files:
        print("⚠️ No .tif or .tiff files found in:", sar_folder)
        exit()

    for fname in files:
        fpath = os.path.join(sar_folder, fname)
        print(f"🔍 Preprocessing: {fname}")
        arr = preprocess_sar_image(fpath)
        print(f"✅ {fname} → shape: {arr.shape}, mean pixel: {arr.mean():.3f}")

        # Save as .npy file
        base_name = os.path.splitext(fname)[0]
        save_path = os.path.join(output_folder, base_name + ".npy")
        np.save(save_path, arr)
        print(f"💾 Saved to: {save_path}")

        # Optional: preview
        plt.imshow(arr, cmap="gray")
        plt.title(fname)
        plt.axis('off')
        plt.show()
