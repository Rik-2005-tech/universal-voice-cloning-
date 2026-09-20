"""Calibrate deepfake-audio detector per language on our own data."""
import io
import os
import wave
import numpy as np
import soundfile as sf

HUMANS = []
for lang in ["hi", "en", "bn"]:
    d = f"data_voice/{lang}"
    files = sorted(f for f in os.listdir(d) if f.endswith(".wav"))[:20]
    HUMANS += [(os.path.join(d, f), lang, 0) for f in files]
HUMANS.append(("test/input/untitled.wav", "bn", 0))

AI = []
from polyvoice.backends import PiperTTS
for lang, model in PiperTTS.VOICE_MAP.items():
    AI.append((model, lang, 1))


def load_any(path, sr_out=16000):
    import soundfile as s2
    x, sr = s2.read(path, always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != sr_out:
        n = int(round(len(x) / sr * sr_out))
        x = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x).astype(np.float32)
    return x


def piper_synth(model, text):
    from piper import PiperVoice
    v = PiperVoice.load(model)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(v.config.sample_rate)
        v.synthesize_wav(text, w)
    buf.seek(0)
    with wave.open(buf, "rb") as r:
        x = (np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16).astype(np.float32)
             / 32768.0)
        sr = r.getframerate()
    if sr != 16000:  # numpy resample (no torchaudio needed)
        n = int(round(len(x) / sr * 16000))
        x = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x).astype(np.float32)
        sr = 16000
    return x, sr


def main():
    import torch
    import argparse
    import sys
    skip_humans = "--ai-only" in sys.argv
    from transformers import pipeline
    texts = {
        "en": "Hello! How are you today? This is a test of the voice detector.",
        "hi": "नमस्ते! आप कैसे हैं? आज मौसम बहुत अच्छा है।",
        "bn": "নমস্কার! আপনি কেমন আছেন? আজ আবহাওয়া খুব ভালো।",
        "fr": "Bonjour, comment vas-tu? Il fait beau aujourd'hui.",
        "es": "Hola, cómo estás? Hace buen tiempo hoy.",
        "zh": "你好，你今天怎么样?今天天气非常好。",
        "ar": "مرحبا، كيف حالك اليوم؟ الطقس جميل جدا.",
    }
    print("loading detector...", flush=True)
    clf = pipeline("audio-classification", model="mo-thecreator/Deepfake-audio-detection",
                   device=-1)
    rows = []
    if skip_humans:
        import json
        with open("logs/deep_calib.json", encoding="utf-8") as fh:
            old = json.load(fh)
        rows = [(r["lang"], r["y"], r["p"], r["src"]) for r in old]
        print(f"loaded {len(rows)} human rows", flush=True)
    else:
        for path, lang, y in HUMANS:
            try:
                x = load_any(path)
                out = clf({"array": x, "sampling_rate": 16000}, top_k=2)
                s = {o["label"]: o["score"] for o in out}
                rows.append((lang, y, float(s.get("fake", 0.0)), os.path.basename(path)))
                print(f"H {lang} p_fake={s.get('fake',0):.3f} {os.path.basename(path)}", flush=True)
            except Exception as e:
                print(f"H {lang} ERR {str(e)[:80]}", flush=True)
    for model, lang, y in AI:
        try:
            x, sr = piper_synth(model, texts[lang])
            out = clf({"array": x, "sampling_rate": sr}, top_k=2)
            s = {o["label"]: o["score"] for o in out}
            rows.append((lang, y, float(s.get("fake", 0.0)), f"piper-{lang}"))
            print(f"A {lang} p_fake={s.get('fake',0):.3f} piper-{lang}", flush=True)
        except Exception as e:
            print(f"A {lang} ERR {str(e)[:80]}", flush=True)
    import json
    with open("logs/deep_calib.json", "w") as fh:
        json.dump([{"lang": l, "y": y, "p": p, "src": s} for l, y, p, s in rows], fh)
    for lang in sorted(set(l for l, _, _, _ in rows)):
        sub = [(y, p) for l, y, p, _ in rows if l == lang]
        for thr in (0.5, 0.3, 0.7):
            acc = sum((p >= thr) == y for y, p in sub) / len(sub)
            print(f"{lang} thr={thr} acc={acc:.3f} n={len(sub)}", flush=True)
    print("CALIB DONE", flush=True)


if __name__ == "__main__":
    main()
