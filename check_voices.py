"""Smoke-test all Piper voices (one native sentence each)."""
import io
import wave
import numpy as np
from piper import PiperVoice

TESTS = [
    ("voices/en-lessac-medium.onnx", "Hello! How are you today?"),
    ("voices/hi-pratham-medium.onnx", "नमस्ते! आप कैसे हैं?"),
    ("voices/bn-google-medium.onnx", "নমস্কার! আপনি কেমন আছেন?"),
    ("voices/fr-siwis-medium.onnx", "Bonjour, comment vas-tu?"),
    ("voices/es-davefx-medium.onnx", "Hola, cómo estás hoy?"),
    ("voices/zh-huayan-medium.onnx", "你好，你今天怎么样?"),
    ("voices/ar-kareem-medium.onnx", "مرحبا، كيف حالك اليوم؟"),
]

for path, text in TESTS:
    v = PiperVoice.load(path)
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
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    print(f"OK {path.split('/')[-1]}: dur={len(x)/sr:.2f}s rms={float((x**2).mean()**0.5):.3f}", flush=True)
print("ALL VOICES OK")
