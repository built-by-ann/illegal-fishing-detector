# make_training_data.py

import os
import numpy as np
import pandas as pd

# Paths
ALIGN_CSV = "data/aligned/sar_ais_pairs.csv"
OUTPUT_DIR = "data/training"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load alignment file
df = pd.read_csv(ALIGN_CSV)

X = []
y = []

# Loop through rows and load SAR images + labels
for idx, row in df.iterrows():
    path = row["sar_path"]
    label = row["fishing"]  # 0 = not fishing, 1 = suspicious

    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        continue

    arr = np.load(path)
    X.append(arr)
    y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Save arrays
np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

print(f"✅ Saved {X.shape[0]} samples to:")
print(f"   → {OUTPUT_DIR}/X.npy")
print(f"   → {OUTPUT_DIR}/y.npy")
