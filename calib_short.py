"""Calibrate short-clip threshold on short HUMANS + short AI (fresh clips)."""
import io
import wave
import numpy as np
import soundfile as sf
from piper import PiperVoice
from polyvoice.spoof import verdict

VOICES = {
    "hi": "voices/hi-pratham-medium.onnx",
    "en": "voices/en-lessac-medium.onnx",
    "bn": "voices/bn-google-medium.onnx",
}
AI_TEXTS = {
    "hi": ["नमस्ते! आप कैसे हैं?", "आज मौसम अच्छा है।", "धन्यवाद, अलविदा!",
           "मेरी ट्रेन छूट गई है।", "आपका नाम क्या है?"],
    "en": ["Hello! How are you?", "Thanks, goodbye!", "What time is it?",
           "I missed my flight.", "Good morning to you."],
    "bn": ["নমস্কার! কেমন আছেন?", "ধন্যবাদ, বিদায়!", "এখন কয়টা বাজে?",
           "আমি ফ্লাইট মিস করেছি।", "আপনার নাম কী?"],
}
HUMAN_IDX = {"hi": [11, 150, 402, 901, 77], "en": [77, 150, 402, 901, 303],
             "bn": [77, 150, 402, 901, 303]}


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
    x = (np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16).astype(np.float32)
         / 32768.0)
    return x, r.getframerate()


def main():
    rows = []  # (p_ai, y)
    for lang, idxs in HUMAN_IDX.items():
        for i in idxs:
            x, sr = sf.read(f"data_voice/{lang}/{lang}_{i:05d}.wav", always_2d=False)
            x = np.asarray(x)
            xs = x[: min(len(x), int(sr * 2.5))]
            v = verdict(xs, sr)
            rows.append((v["p_ai"], 0, f"{lang}-humshort-{i}"))
            print(f"H {lang}_{i} (2.5s): p={v['p_ai']}", flush=True)
    for lang, ts in AI_TEXTS.items():
        for t in ts:
            x, sr = psynth(VOICES[lang], t)
            v = verdict(x, sr)
            rows.append((v["p_ai"], 1, f"{lang}-aishort"))
            print(f"A {lang} ({len(x)/sr:.1f}s): p={v['p_ai']}", flush=True)
    best_t, best_a = 0.5, -1.0
    t = 0.05
    while t < 1.0:
        acc = sum(((p >= t) == y) for p, y, _ in rows) / len(rows)
        fa = sum((p >= t) and y == 0 for p, y, _ in rows)
        miss = sum((p < t) and y == 1 for p, y, _ in rows)
        print(f"thr={t:.2f} acc={acc:.3f} false-alarm={fa} miss={miss}", flush=True)
        if acc > best_a:
            best_a, best_t = acc, t
        t += 0.05
    print(f"BEST_SHORT_THR={best_t:.2f} acc={best_a:.3f} n={len(rows)}", flush=True)
    # human-side safety: max human score must stay below threshold
    hm = max(p for p, y, _ in rows if y == 0)
    print(f"max short-human p_ai={hm:.3f}", flush=True)


if __name__ == "__main__":
    main()
