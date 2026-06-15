
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

WIN = 16000 * 4  # the model expects 4 s at 16 kHz

def _loudest_window(y):
    """Return the 4-second slice with the most energy. A recording with a silent
    lead-in (click mic, breathe, then speak) or one longer than 4 s would
    otherwise be truncated to the first 4 s and read as silence."""
    if y.size <= WIN:
        return y
    # O(n) sliding-window energy via a cumulative sum of squares
    csum = np.cumsum(np.concatenate(([0.0], y.astype(np.float64) ** 2)))
    hop = 1600  # 0.1 s steps
    best_start, best_e = 0, -1.0
    for start in range(0, y.size - WIN + 1, hop):
        e = csum[start + WIN] - csum[start]
        if e > best_e:
            best_e, best_start = e, start
    return y[best_start:best_start + WIN]

def predict(file_path):
    # Load the WHOLE clip (no 4 s cap), then score the loudest 4 s window — so a
    # quiet lead-in or a long recording isn't silently truncated to nothing.
    audio, sr = librosa.load(file_path, sr=16000)
    dur = (len(audio) / sr) if audio.size else 0.0
    seg = _loudest_window(audio)
    rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=40)
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
    st.caption(f"Decoded clip: {dur:.1f}s · 16 kHz · loudest-window signal (RMS) = {rms:.4f}")
    if rms == 0.0:
        st.error(
            "🔇 A completely silent track was recorded (all-zero samples) — the mic "
            "audio never reached the browser. This is a device/OS issue, not your "
            "speech, so there's nothing to analyze. Try, in order:\n\n"
            "1. **Windows → Privacy & security → Microphone** — turn on mic access for "
            "apps / desktop apps.\n"
            "2. **Settings → System → Sound → Input** — select your real mic (not "
            "'Stereo Mix' / a virtual device) and raise the level.\n"
            "3. Click the **🎤 icon in the browser address bar** and pick the correct mic.\n"
            "4. Use **Chrome or Edge** — Brave and iOS Safari can record silence."
        )
        return
    if rms < SILENCE_RMS:
        st.warning(
            "⚠️ This clip is very quiet (low signal level above). Speak louder/closer to "
            "the mic, or check your input device — the result below may be unreliable."
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
    st.caption("Click the mic, **speak right away**, then click it again to stop.")
    st.info(
        "🎤 Allow the microphone prompt when it appears. If the result comes back "
        "silent, it's almost always the input device — check your OS mic settings and "
        "the 🎤 icon in the address bar, and use Chrome/Edge. "
        "(Running locally? Use **http://localhost**, not a `192.168.x.x` URL — browsers "
        "block the mic on plain-http LAN addresses.)"
    )
    recorded = st.audio_input("Record audio")
    if recorded:
        analyze(recorded)
