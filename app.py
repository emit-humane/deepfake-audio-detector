
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root{
  --card:rgba(255,255,255,.045); --stroke:rgba(255,255,255,.10);
  --txt:#e6e9f0; --muted:#9aa3b8;
  --grad:linear-gradient(135deg,#22d3ee 0%,#7c5cff 50%,#f0abfc 100%);
}

.stApp{
  background:
    radial-gradient(1100px 520px at 12% -12%, rgba(124,92,255,.22), transparent 60%),
    radial-gradient(950px 520px at 112% 6%, rgba(34,211,238,.16), transparent 55%),
    #0b0f1a;
  color:var(--txt); font-family:'Inter',sans-serif;
}
#MainMenu, header, footer{visibility:hidden;}
.block-container{padding-top:2.4rem; max-width:840px;}

/* hero */
.hero{ text-align:center; margin-bottom:1.1rem; }
.hero .badge{
  display:inline-block; font-size:.7rem; letter-spacing:.2em; text-transform:uppercase;
  color:var(--muted); border:1px solid var(--stroke); border-radius:999px;
  padding:.34rem .85rem; margin-bottom:1rem; background:var(--card); backdrop-filter:blur(8px);
}
.hero h1{
  font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:3.2rem; line-height:1.04; margin:.1rem 0;
  background:var(--grad); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  background-size:200% auto; animation:shine 6s linear infinite;
  filter:drop-shadow(0 8px 34px rgba(124,92,255,.35));
}
@keyframes shine{ to{ background-position:200% center; } }
.hero p{ color:var(--muted); font-size:1.06rem; margin:.45rem auto 0; max-width:520px; }

/* stat cards */
.metrics{ display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; margin:1.4rem 0 .4rem; }
.metric-card{
  background:var(--card); border:1px solid var(--stroke); border-radius:16px;
  padding:1.05rem .5rem; text-align:center; backdrop-filter:blur(10px);
  transition:transform .2s ease, border-color .2s ease, box-shadow .2s ease;
}
.metric-card:hover{ transform:translateY(-4px); border-color:rgba(124,92,255,.55);
  box-shadow:0 16px 40px -18px rgba(124,92,255,.6); }
.metric-card .v{ font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700;
  background:var(--grad); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.metric-card .l{ color:var(--muted); font-size:.7rem; letter-spacing:.09em; text-transform:uppercase; margin-top:.3rem; }
@media(max-width:640px){ .metrics{ grid-template-columns:repeat(2,1fr);} .hero h1{font-size:2.4rem;} }

/* tabs */
.stTabs [data-baseweb="tab-list"]{ gap:.5rem; border-bottom:1px solid var(--stroke); }
.stTabs [data-baseweb="tab"]{
  background:var(--card); border:1px solid var(--stroke); border-bottom:none;
  border-radius:12px 12px 0 0; padding:.55rem 1.15rem; color:var(--muted); font-weight:600;
}
.stTabs [aria-selected="true"]{ color:#fff; border-color:rgba(124,92,255,.6);
  background:linear-gradient(180deg, rgba(124,92,255,.24), rgba(124,92,255,.03)); }

/* uploader / inputs */
[data-testid="stFileUploaderDropzone"], [data-testid="stAudioInput"]{
  background:var(--card)!important; border:1.5px dashed var(--stroke)!important; border-radius:16px!important;
}
[data-testid="stFileUploaderDropzone"]:hover{ border-color:rgba(124,92,255,.6)!important; }

/* result card */
.result{ border-radius:20px; padding:1.4rem 1.2rem; text-align:center; color:#fff;
  font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700; margin:1.1rem 0 .5rem;
  animation:pop .45s cubic-bezier(.2,.9,.3,1.3); }
.result span{ display:block; font-family:'Inter',sans-serif; font-weight:500; font-size:.88rem; opacity:.92; margin-top:.35rem; }
.result-fake{ background:linear-gradient(135deg,#ff416c,#ff4b2b); box-shadow:0 18px 52px -14px rgba(255,65,108,.6); }
.result-real{ background:linear-gradient(135deg,#11998e,#38ef7d); box-shadow:0 18px 52px -14px rgba(17,153,142,.6); }
@keyframes pop{ from{opacity:0; transform:scale(.93) translateY(8px);} to{opacity:1; transform:none;} }

/* result metrics + progress */
[data-testid="stMetric"]{ background:var(--card); border:1px solid var(--stroke); border-radius:14px; padding:.75rem 1rem; }
[data-testid="stMetricValue"]{ font-family:'Space Grotesk',sans-serif; }
.stProgress > div > div > div > div{ background:var(--grad)!important; }
audio{ width:100%; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="badge">CRNN · CNN + BiLSTM</div>
  <h1>Deepfake Audio Detector</h1>
  <p>Is this voice a real human — or AI-generated? Upload a clip or record live to find out.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="metrics">
  <div class="metric-card"><div class="v">97.89%</div><div class="l">Accuracy</div></div>
  <div class="metric-card"><div class="v">97.89%</div><div class="l">F1 Score</div></div>
  <div class="metric-card"><div class="v">1.47%</div><div class="l">EER</div></div>
  <div class="metric-card"><div class="v">CRNN</div><div class="l">Model</div></div>
</div>
""", unsafe_allow_html=True)

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
        st.markdown(
            '<div class="result result-fake">🚨 FAKE — AI Generated'
            '<span>This voice looks synthetically generated</span></div>',
            unsafe_allow_html=True)
    else:
        confidence = (1 - prob) * 100
        st.markdown(
            '<div class="result result-real">✅ REAL — Human Voice'
            '<span>This voice looks like a genuine human</span></div>',
            unsafe_allow_html=True)
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
