"""All-language random test: hi/en/bn/fr/es, fresh clips + fresh AI texts."""
import io
import random
import wave
import numpy as np
import soundfile as sf
from piper import PiperVoice
from polyvoice.spoof import verdict, phone_channel

rng = random.Random(99)
HUMAN_POOL = {
    "hi": [111, 222, 333, 444, 555],
    "en": [111, 222, 333, 444, 555],
    "bn": [111, 222, 333, 444, 555],
    "fr": "__world__",
    "es": "__world__",
}
WORLD_POOL = {}
for _lang in ["fr", "es"]:
    d = f"data_voice_world/{_lang}"
    import os
    files = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
    rng.shuffle(files)
    WORLD_POOL[_lang] = [os.path.join(d, f) for f in files[:5]]
AI_TEXTS = {
    "hi": ["आज बाज़ार में भीड़ है।", "यह चाय बहुत गरम है।", "कल छुट्टी है।"],
    "en": ["The market is busy today.", "This tea is very hot.", "Tomorrow is a holiday."],
    "bn": ["আজ বাজারে ভিড় আছে।", "এই চা খুব গরম।", "আগামীকাল ছুটি।"],
    "fr": ["Le marché est animé aujourd'hui.", "Ce thé est très chaud.", "Demain c'est férié."],
    "es": ["El mercado está concurrido hoy.", "Este té está muy caliente.", "Mañana es fiesta."],
}
VOICES = {
    "hi": "voices/hi-pratham-medium.onnx",
    "en": "voices/en-lessac-medium.onnx",
    "bn": "voices/bn-google-medium.onnx",
    "fr": "voices/fr-siwis-medium.onnx",
    "es": "voices/es-davefx-medium.onnx",
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
    x = (np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16).astype(np.float32)
         / 32768.0)
    return x, r.getframerate()


def main():
    from polyvoice.backends import SuperhumanTTS
    cases = []
    for lang, idxs in HUMAN_POOL.items():
        if idxs == "__world__":
            continue
        for i in idxs:
            x, sr = sf.read(f"data_voice/{lang}/{lang}_{i:05d}.wav", always_2d=False)
            x = np.asarray(x)
            cases.append((f"{lang}-HUMAN", x, sr, "HUMAN", lang))
            cases.append((f"{lang}-HUMAN-phone",
                          phone_channel(x, sr, seed=i), sr, "HUMAN", lang))
    for lang, files in WORLD_POOL.items():
        for f in files:
            x, sr = sf.read(f, always_2d=False)
            x = np.asarray(x)
            cases.append((f"{lang}-HUMAN", x, sr, "HUMAN", lang))
    for lang, m in VOICES.items():
        for t in AI_TEXTS[lang]:
            x, sr = psynth(m, t)
            cases.append((f"{lang}-AI", x, sr, "AI", lang))
    tts = SuperhumanTTS(ckpt="bank_large.npz")
    for lang, t in [("en", "Final round check"), ("hi", "आखिरी जांच"),
                    ("bn", "শেষ পরীক্ষা"), ("fr", "Contrôle final"),
                    ("es", "Control final")]:
        w, s = tts.synth(t, lang)
        cases.append((f"{lang}-hum", np.asarray(w), s, "AI", lang))
    ok = 0
    per = {}
    for name, x, sr, want, lang in cases:
        v = verdict(x, sr, lang=lang)
        good = v["label"] == want
        ok += good
        per.setdefault(lang, [0, 0])[good] += 1
        print(f"{name}: -> {v['label']} (p={v['p_ai']}, m={v.get('model')}) "
              f"{'OK' if good else 'MISS'}", flush=True)
    print(f"ALL-LANG {ok}/{len(cases)} = {ok/len(cases):.3f}", flush=True)
    for lang, (g, b) in sorted(per.items()):
        print(f"  {lang}: {g} ok / {b} miss", flush=True)


if __name__ == "__main__":
    main()
