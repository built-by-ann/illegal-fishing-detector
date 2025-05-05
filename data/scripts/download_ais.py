# data/scripts/download_ais.py

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("GFW_API_TOKEN")

headers = {
    "Authorization": f"Bearer {TOKEN}"
}

# Define bounding box and time range
bbox = [-80.0, -10.0, -79.0, -9.0]  # off the coast of Ecuador (Galápagos)
start_date = "2023-06-01T00:00:00Z"
end_date = "2023-06-02T00:00:00Z"

# Vessel search endpoint
url = "https://gateway.api.globalfishingwatch.org/vessels"

params = {
    "start": start_date,
    "end": end_date,
    "bbox": ",".join(map(str, bbox)),
    "limit": 100
}

print("Querying GFW API...")
response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    print("✅ Data received")
    data = response.json()
    if data:
        df = pd.DataFrame(data)
        os.makedirs("data/raw/ais", exist_ok=True)
        df.to_csv("data/raw/ais/sample_ais.csv", index=False)
        print("✅ Saved to data/raw/ais/sample_ais.csv")
    else:
        print("⚠️ No data returned")
else:
    print(f"❌ Error {response.status_code}: {response.text}")
