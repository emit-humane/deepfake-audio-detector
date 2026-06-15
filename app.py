
# --- quiet the noisy (harmless) TensorFlow / librosa startup logs ---
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # hide oneDNN / CPU-feature notices
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import warnings
warnings.filterwarnings("ignore")                     # incl. librosa "PySoundFile failed" fallback
import logging
for _n in ("tensorflow", "absl"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import tempfile

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="centered")

st.markdown("""
<style>
    .title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        background: linear-gradient(90deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle { text-align: center; color: #888; font-size: 1.1em; margin-bottom: 2em; }
    .result-fake {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        padding: 20px; border-radius: 15px; text-align: center;
        font-size: 1.8em; font-weight: bold; color: white; margin: 20px 0;
    }
    .result-real {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 20px; border-radius: 15px; text-align: center;
        font-size: 1.8em; font-weight: bold; color: white; margin: 20px 0;
    }
    .record-btn {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white; border: none; padding: 15px 40px;
        border-radius: 50px; font-size: 1.2em;
        cursor: pointer; margin: 10px;
        transition: all 0.3s;
    }
    .stop-btn {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white; border: none; padding: 15px 40px;
        border-radius: 50px; font-size: 1.2em;
        cursor: pointer; margin: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🎙️ Deepfake Audio Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect if audio is Real (Human) or Fake (AI Generated)</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "97.89%")
col2.metric("F1 Score", "97.89%")
col3.metric("EER", "1.47%")
col4.metric("Model", "CRNN")

st.divider()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5", compile=False)

model = load_model()

# Below this RMS the clip is effectively silent. The model maps silence to a
# confident FAKE (prob ~1.0), so we surface a caution when a clip decodes this
# quiet — but we no longer block the result, so a real clip always gets scored.
SILENCE_RMS = 0.005

ALLOWED_TYPES = ["wav", "mp3", "flac", "m4a", "ogg", "aac"]

def predict(file_path):
    audio, sr = librosa.load(file_path, sr=16000, duration=4.0)
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    dur = (len(audio) / sr) if audio.size else 0.0
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 200:
        mfcc = np.pad(mfcc, ((0,0),(0, 200 - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :200]
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    prob = model.predict(mfcc, verbose=0)[0][0]
    return prob, rms, dur

def show_result(prob):
    if prob > 0.5:
        confidence = prob * 100
        st.markdown('<div class="result-fake">🚨 FAKE — AI Generated</div>', unsafe_allow_html=True)
    else:
        confidence = (1 - prob) * 100
        st.markdown('<div class="result-real">✅ REAL — Human Voice</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{confidence:.2f}%")
    c2.metric("Fake Probability", f"{prob*100:.2f}%")
    st.progress(float(prob))

def analyze(uploaded):
    """Play, run the model on, and report a result for an uploaded/recorded clip."""
    st.audio(uploaded)
    data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
    if not data:
        st.error("This clip is empty (0 bytes) — nothing was recorded or uploaded.")
        return
    suffix = os.path.splitext(getattr(uploaded, "name", "") or "")[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        with st.spinner("🔍 Analyzing..."):
            prob, rms, dur = predict(tmp_path)
    except Exception as e:
        st.error(f"Couldn't decode this audio file ({suffix}): {e}")
        return
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    # Always show what was actually decoded, so a silent capture is visible.
    st.caption(f"Decoded clip: {dur:.1f}s · 16 kHz · signal level (RMS) = {rms:.4f}")
    if rms < SILENCE_RMS:
        st.warning(
            "⚠️ This clip decoded to almost no sound (see the very low signal level "
            "above). If you were speaking, the audio isn't reaching the app — check "
            "the browser's microphone permission and that the right input device is "
            "selected. The result below is unreliable for near-silent audio "
            "(the model tends to output FAKE)."
        )
    show_result(prob)

tab1, tab2 = st.tabs(["📁 Upload Audio", "🎙️ Live Record"])

with tab1:
    st.markdown("### Upload an audio file")
    uploaded_file = st.file_uploader("WAV, MP3, FLAC, M4A, OGG, AAC supported", type=ALLOWED_TYPES)
    if uploaded_file:
        analyze(uploaded_file)

with tab2:
    st.markdown("### Record directly from your microphone 🎙️")
    st.caption("Click the mic, speak for a few seconds, then click it again to stop.")
    st.info(
        "🔒 Microphone only works on a **secure origin**. Open this app at "
        "**http://localhost:8501** — not the `http://192.168.x.x` network URL. "
        "Browsers block mic access on plain-http LAN addresses, which records silence."
    )
    recorded = st.audio_input("Record audio")
    if recorded:
        analyze(recorded)
