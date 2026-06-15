---
title: Deepfake Audio Detector
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.46.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🎙️ Deepfake Audio Detector — CRNN (CNN + BiLSTM)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-97.89%25-green)
![Model](https://img.shields.io/badge/Model-CRNN-purple)

## 📌 Project Description
A deep learning system that classifies speech recordings as **Real (Human)** or **Fake (AI Generated)** using a **CRNN (Convolutional Recurrent Neural Network)** on MFCC features. A CNN front-end extracts local time–frequency patterns; a Bidirectional LSTM back-end models how the signal evolves over time — the cue that most reliably separates synthetic speech from real.

---

## 🏆 Results

| Metric | CNN Baseline | **CRNN (This Project)** | Required | Status |
|--------|:------------:|:-----------------------:|:--------:|:------:|
| Accuracy | 90.74% | **97.89%** | 80% | ✅ PASS |
| F1 Score | 90.10% | **97.89%** | 80% | ✅ PASS |
| EER | 3.17% | **1.47%** | 12% | ✅ PASS |
| Real Recall | 99.5% | **100%** | 75% | ✅ PASS |
| Fake Recall | 82.4% | **96%** | 75% | ✅ PASS |

The CRNN outperforms the CNN baseline across every metric. EER dropped from 3.17% → 1.47%, and fake recall improved from 82.4% → 96% — the hardest class to get right.

---

## 📂 Dataset
- **Dataset:** Fake-or-Real (Kaggle) — `mohammedabdeldayem/the-fake-or-real-dataset`
- **Folder:** `for-norm`
- **Training:** 53,868 files · **Validation:** 10,798 files · **Testing:** 4,634 files

---

## 🧠 Methodology

### 1. Preprocessing
- Audio loaded at 16 kHz, clipped to 4 seconds
- Converted to MFCC features — 40 coefficients × 200 frames

### 2. Feature Extraction
- MFCC (Mel Frequency Cepstral Coefficients)
- Each audio file → 40 × 200 grid

### 3. Model Architecture — CRNN

**CNN front-end** (local pattern extraction)
- Input (40, 200, 1)
- Conv2D(32) + BatchNorm + MaxPool(2,2) + Dropout(0.25) → (20, 100, 32)
- Conv2D(64) + BatchNorm + MaxPool(2,2) + Dropout(0.25) → (10, 50, 64)
- Conv2D(128) + BatchNorm + MaxPool(2,2) + Dropout(0.30) → (5, 25, 128)

**Bridge**
- Permute → time axis first → Reshape to (time=25, features=640)

**Recurrent back-end** (temporal dynamics)
- Bidirectional LSTM(128, return_sequences=True) + Dropout(0.30)
- Bidirectional LSTM(64) + Dropout(0.40)

**Classifier head**
- Dense(64, relu) + Dropout(0.30)
- Dense(1, sigmoid) — 0 = Real, 1 = Fake

**Training config:** ~1.05M parameters · Optimizer: Adam · Loss: Binary Crossentropy · Epochs: 20 (best at epoch 18) · Batch size: 64

---

## 📊 Confusion Matrix (Test set — 4,634 samples)

| | Predicted Real | Predicted Fake |
|---|:---:|:---:|
| **Actual Real** | 2,264 ✅ | 0 ❌ |
| **Actual Fake** | 97 ❌ | 2,273 ✅ |

---

## 📁 Project Structure
```
deepfake-audio-detector/
├── colab_notebook_crnn.ipynb   ← Full training notebook (run on Colab GPU)
├── app.py                      ← Streamlit web app
├── predict.py                  ← CLI prediction script
├── best_model.h5               ← Trained CRNN model (epoch 18)
└── report/
    ├── confusion_matrix.png
    └── training_history.png
```

---

## 🚀 How to Run

**Install dependencies:**
```bash
pip install tensorflow librosa streamlit scikit-learn
```

**Run the web app:**
```bash
streamlit run app.py
```

**Test a single file (CLI):**
```bash
python predict.py your_audio.wav
```

**Train from scratch (Google Colab):**
1. Open `colab_notebook_crnn.ipynb` in Colab
2. **Runtime → Change runtime type → GPU**
3. Run all cells top to bottom

---

## 🌐 Web App Features
- **Upload Audio:** WAV / MP3 / FLAC supported
- **Live Record:** Record from browser mic
- Shows Real / Fake result with confidence score and fake probability bar

---

## 👨‍💻 Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| TensorFlow / Keras | CRNN model |
| Librosa | Audio processing |
| Scikit-learn | Metrics |
| Streamlit | Web app |
| Google Colab | GPU training |
| ngrok | Public URL for deployment |
