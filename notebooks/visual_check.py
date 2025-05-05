# notebooks/visual_check.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the alignment CSV
df = pd.read_csv("data/aligned/sar_ais_pairs.csv")

# Pick the first match
sample = df.iloc[0]

# Load the corresponding SAR image (.npy)
img = np.load(sample["sar_path"])

# Show info
print("🛥️ Vessel:", sample["vessel_name"])
print("📅 Timestamp:", sample["timestamp"])
print("🎯 Fishing Label:", sample["fishing"])
print("📍 Location:", (sample["lat"], sample["lon"]))

# Display the image
plt.imshow(img, cmap="gray")
plt.title(f"{sample['vessel_name']} (Fishing: {sample['fishing']})")
plt.axis("off")
plt.show()
