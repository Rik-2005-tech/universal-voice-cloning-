"""Round-trip: Piper speaks hi/bn -> faster-whisper detects+transcribes."""
import io
import wave
import numpy as np
import soundfile as sf
from piper import PiperVoice

CASES = [
    ("voices/hi-pratham-medium.onnx", "नमस्ते! आप कैसे हैं?", "/tmp/real_hi.wav"),
    ("voices/bn-google-medium.onnx", "নমস্কার! আপনি কেমন আছেন?", "/tmp/real_bn.wav"),
    ("voices/en-lessac-medium.onnx", "Hello! How are you today?", "/tmp/real_en.wav"),
]

for model, text, out in CASES:
    v = PiperVoice.load(model)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(v.config.sample_rate)
        v.synthesize_wav(text, w)
    buf.seek(0)
    with wave.open(buf, "rb") as r:
        raw = r.readframes(r.getnframes())
        sr = r.getframerate()
    x = (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)
    # resample to 16k for whisper
    n_out = int(round(len(x) / sr * 16000))
    x16 = np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, len(x)), x).astype(np.float32)
    sf.write(out, x16, 16000)
    print(f"made {out} dur={len(x16)/16000:.2f}s", flush=True)

from faster_whisper import WhisperModel
print("loading whisper tiny...", flush=True)
m = WhisperModel("tiny", device="cpu", compute_type="int8")
for out in ["/tmp/real_hi.wav", "/tmp/real_bn.wav", "/tmp/real_en.wav"]:
    x, sr = sf.read(out)
    segs, info = m.transcribe(np.asarray(x, dtype=np.float32), beam_size=1, vad_filter=True)
    txt = " ".join(s.text.strip() for s in segs).strip()
    print(f"{out}: lang={info.language} p={info.language_probability:.2f} text={txt!r}", flush=True)
print("ROUND TRIP DONE")
