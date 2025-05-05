# utils/align_data.py

import os
import pandas as pd
from datetime import datetime
import numpy as np
import re

# Paths
SAR_FOLDER = "data/preprocessed/sar"
AIS_PATH = "data/raw/ais/mock_ais.csv"
OUTPUT_PATH = "data/aligned/sar_ais_pairs.csv"
os.makedirs("data/aligned", exist_ok=True)

# Load AIS data
ais_df = pd.read_csv(AIS_PATH)
ais_df["timestamp"] = pd.to_datetime(ais_df["timestamp"]).dt.tz_localize(None)

results = []

for fname in os.listdir(SAR_FOLDER):
    if not fname.endswith(".npy"):
        continue

    # 🧠 Extract timestamp using regex (pattern: 20230629t115022)
    match_ts = re.search(r"\d{8}t\d{6}", fname.lower())
    if match_ts:
        timestamp_str = match_ts.group()
        sar_time = datetime.strptime(timestamp_str, "%Y%m%dt%H%M%S")
    else:
        print(f"❌ Could not extract timestamp from: {fname}")
        continue

    # Find closest AIS row
    ais_df["time_diff"] = (ais_df["timestamp"] - sar_time).abs()
    match = ais_df.sort_values("time_diff").iloc[0]

    print(f"🔗 Matched {fname} ←→ {match['vessel_name']} at {match['timestamp']}")

    results.append({
        "sar_path": os.path.join(SAR_FOLDER, fname),
        "mmsi": match["mmsi"],
        "vessel_name": match["vessel_name"],
        "timestamp": match["timestamp"],
        "lat": match["lat"],
        "lon": match["lon"],
        "fishing": match["fishing"]
    })

# Save aligned data
matched_df = pd.DataFrame(results)
matched_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved aligned pairs to {OUTPUT_PATH}")
