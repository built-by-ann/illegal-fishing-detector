# 🐟 illegal-fishing-detector

An end-to-end system to detect illegal fishing activity using satellite radar (SAR) imagery and vessel tracking data.

---

## 📌 Project Overview

Illegal, unreported, and unregulated (IUU) fishing costs the global economy billions annually and endangers marine ecosystems. Many IUU vessels turn off their AIS (Automatic Identification System) to evade detection — but **Synthetic Aperture Radar (SAR)** satellites can still "see" them regardless of weather or lighting.

This project builds a full-stack prototype system that can:

- Load and preprocess SAR satellite images
- Align them with (mock) AIS vessel data
- Train a deep learning classifier to detect fishing activity
- Visualize predictions and vessel positions
- (Future) Analyze vessel behavior over time using LSTM models

---

## 🎯 Goals

- 🛰️ Detect likely fishing vessels from raw satellite imagery
- 🧠 Train a CNN to distinguish fishing vs. non-fishing activity
- 🗂️ Align radar data with vessel tracks (AIS)
- 🔮 Build a usable prediction pipeline
- 📈 Provide a foundation for future LSTM-based movement analysis

---

## 🧪 Technical Stack

| Layer        | Tech                             |
|--------------|----------------------------------|
| ML Model     | TensorFlow / Keras (CNN)         |
| Data         | Sentinel-1 SAR + mock AIS (CSV)  |
| Preprocessing| NumPy, Pillow, Pandas            |
| API (WIP)    | Flask (Planned)                  |
| Frontend (WIP)| React + Mapbox (Planned)        |

---

## 🔄 Pipeline Progress

### ✅ SAR Data
- Downloaded Sentinel-1 SAR `.tiff` files from Galápagos region
- Preprocessed to grayscale, resized to `224x224`, saved as `.npy`

### ✅ AIS Data (Mock)
- Created mock AIS CSV with timestamp, position, MMSI, and fishing label
- Aligned SAR images to closest AIS rows by timestamp

### ✅ Training Dataset
- Constructed paired dataset: `X.npy` (images), `y.npy` (labels)
- Balanced classes (fishing vs. non-fishing)

### ✅ CNN Model
- Trained a Convolutional Neural Network classifier on SAR images
- Achieved 100% accuracy on small balanced dataset (due to limited scope + mock data)

### 🔜 LSTM Behavior Module
- Placeholder created for vessel movement analysis using AIS track sequences
- To be implemented in a future version (e.g. LSTM over lat/lon time series)

---

## 📁 Project Structure

```bash
illegal-fishing-detector/
├── api/                    # Flask API (in progress)
├── data/
│   ├── raw/                # Raw SAR & AIS data
│   ├── preprocessed/       # .npy SAR arrays
│   └── training/           # Final training X/y files
├── docs/                   # Architecture, blog drafts
├── frontend/               # React UI (in progress)
├── model/
│   ├── cnn_model.py
│   ├── lstm_sequence.py    # LSTM stub (future)
│   └── train_model.py      # CNN trainer
├── notebooks/              # EDA + experiments
├── utils/
│   ├── preprocess_sar.py
│   ├── make_training_data.py
│   └── align_data.py
├── .env                    # API keys (ignored)
├── .gitignore
├── README.md               # ← You are here
```

---

## 📊 Example Output

- CNN Accuracy: **100%** (on 12 SAR images + mock AIS labels)
- Balanced classes, low validation loss
- Sample prediction: ✅ Likely fishing detected for “Fake Tuna One” at 2023-06-29 11:50 UTC

---

## 🚧 To-Do / Roadmap

- [ ] 🌊 Add real AIS sequences via Global Fishing Watch API
- [ ] 🧠 Train LSTM model on vessel track behavior
- [ ] 🛰️ Fuse AIS + SAR for multi-modal classification
- [ ] 🧪 Evaluate on larger/more realistic datasets
- [ ] 🖥️ Build interactive web viewer (Mapbox, Flask)
- [ ] 📤 Deploy prediction API and UI on Hugging Face Spaces or Streamlit

---

## 📖 References

- [Sentinel-1 SAR Data Hub](https://scihub.copernicus.eu/)
- [Global Fishing Watch API](https://globalfishingwatch.org/data-download/)
- [TensorFlow CNN Documentation](https://www.tensorflow.org/tutorials/images/cnn)

---

## 🧑‍💻 Author

Ann Mathew — AI + oceans + software  
Made with 💙 for curious conservationists and ML enthusiasts!
