"""Full scenario test: hi/en/bn x (human, AI, phone-human, phone-AI, short-AI)."""
import io
import wave
import numpy as np
import soundfile as sf
from piper import PiperVoice
from polyvoice.spoof import verdict, phone_channel

VOICES = {
    "hi": ("voices/hi-pratham-medium.onnx", "मेरी ट्रेन छूट गई है, अब मैं क्या करूं?"),
    "en": ("voices/en-lessac-medium.onnx", "I missed my flight, what should I do now?"),
    "bn": ("voices/bn-google-medium.onnx", "আমি আমার ফ্লাইট মিস করেছি, এখন কী করব?"),
}
SHORT = {
    "hi": ("voices/hi-pratham-medium.onnx", "नमस्ते! आप कैसे हैं?"),
    "en": ("voices/en-lessac-medium.onnx", "Hello! How are you?"),
    "bn": ("voices/bn-google-medium.onnx", "নমস্কার! কেমন আছেন?"),
}


def psynth(model, text):
    v = PiperVoice.load(model)
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(v.config.sample_rate)
    v.synthesize_wav(text, w)
    w.close()
    buf.seek(0)
    r = wave.open(buf, "rb")
    x = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return x, r.getframerate()


def main():
    cases = []
    for lang, idx in [("hi", [5, 303]), ("en", [11, 611]), ("bn", [9, 702])]:
        for i in idx:
            x, sr = sf.read(f"data_voice/{lang}/{lang}_{i:05d}.wav", always_2d=False)
            x = np.asarray(x)
            cases.append((f"{lang}-HUMAN", x, sr, "HUMAN"))
            cases.append((f"{lang}-HUMAN-phone", phone_channel(x, sr, seed=i), sr, "HUMAN"))
    for lang, (m, t) in VOICES.items():
        x, sr = psynth(m, t)
        cases.append((f"{lang}-AI", x, sr, "AI"))
        cases.append((f"{lang}-AI-phone", phone_channel(x, sr, seed=3), sr, "AI"))
    for lang, (m, t) in SHORT.items():
        x, sr = psynth(m, t)
        cases.append((f"{lang}-AI-short", x, sr, "AI"))
    ok = 0
    per_lang = {}
    for name, x, sr, want in cases:
        v = verdict(x, sr)
        good = v["label"] == want
        ok += good
        lang = name.split("-")[0]
        per_lang.setdefault(lang, [0, 0])
        per_lang[lang][good] += 1
        print(f"{name}: -> {v['label']} (p_ai={v['p_ai']}) {'OK' if good else 'MISS'}", flush=True)
    print(f"TOTAL {ok}/{len(cases)}", flush=True)
    for lang, (g, b) in sorted(per_lang.items()):
        print(f"{lang}: {g} ok / {b} miss", flush=True)


if __name__ == "__main__":
    main()
