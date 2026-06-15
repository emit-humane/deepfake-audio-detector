
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import warnings
warnings.filterwarnings("ignore")
import logging
for _n in ("tensorflow", "absl"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import numpy as np
import librosa
import tensorflow as tf
import sys

# Model load karo
model = tf.keras.models.load_model('best_model.h5', compile=False)

WIN = 16000 * 4  # the model expects 4 s at 16 kHz

def _loudest_window(y):
    """Pick the 4 s slice with the most energy so a silent lead-in or a long
    recording isn't truncated to the (possibly quiet) first 4 s."""
    if y.size <= WIN:
        return y
    csum = np.cumsum(np.concatenate(([0.0], y.astype(np.float64) ** 2)))
    best_start, best_e = 0, -1.0
    for start in range(0, y.size - WIN + 1, 1600):
        e = csum[start + WIN] - csum[start]
        if e > best_e:
            best_e, best_start = e, start
    return y[best_start:best_start + WIN]

def predict_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    audio = _loudest_window(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 200:
        mfcc = np.pad(mfcc, ((0,0),(0, 200 - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :200]

    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    prob = model.predict(mfcc, verbose=0)[0][0]

    label = "FAKE (AI Generated)" if prob > 0.5 else "REAL (Human)"
    confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100

    print(f"Result     : {label}")
    print(f"Confidence : {confidence:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file>")
        sys.exit(1)
    file_path = sys.argv[1]
    predict_audio(file_path)
