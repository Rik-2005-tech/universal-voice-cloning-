"""Random test set v2: fresh seeds, indices and texts never used anywhere."""
import io
import random
import wave
import numpy as np
import soundfile as sf
from piper import PiperVoice
from polyvoice.spoof import verdict, phone_channel

rng = random.Random(21)
POOL = {"hi": [66, 250, 450, 650, 850], "en": [120, 260, 470, 680, 880],
        "bn": [120, 260, 470, 680, 880]}
VOICES = {
    "hi": "voices/hi-pratham-medium.onnx",
    "en": "voices/en-lessac-medium.onnx",
    "bn": "voices/bn-google-medium.onnx",
}
AI_TEXTS = {
    "hi": ["सुबह की सैर स्वास्थ्य के लिए अच्छी है।", "कल रात बहुत बारिश हुई।",
           "यह आम बहुत मीठा है।", "कृपया दरवाज़ा बंद कर दें।",
           "हम अगले हफ्ते गाँव जाएँगे।"],
    "en": ["Morning walks are good for health.", "It rained heavily last night.",
           "This mango is very sweet.", "Please close the door.",
           "We will visit the village next week."],
    "bn": ["সকালের হাঁটা স্বাস্থ্যের জন্য ভালো।", "গতরাতে অনেক বৃষ্টি হয়েছে।",
           "এই আমটা খুব মিষ্টি।", "দয়া করে দরজা বন্ধ করুন।",
           "আমরা আগামী সপ্তাহে গ্রামে যাব।"],
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
    for lang, idxs in POOL.items():
        for i in idxs:
            try:
                x, sr = sf.read(f"data_voice/{lang}/{lang}_{i:05d}.wav", always_2d=False)
            except Exception as e:
                print(f"skip {lang}_{i}: {str(e)[:40]}", flush=True)
                continue
            x = np.asarray(x)
            cases.append((f"{lang}-HUMAN", x, sr, "HUMAN"))
            cases.append((f"{lang}-HUMAN-phone", phone_channel(x, sr, seed=i), sr, "HUMAN"))
    for lang, m in VOICES.items():
        for t in AI_TEXTS[lang]:
            x, sr = psynth(m, t)
            cases.append((f"{lang}-AI", x, sr, "AI"))
    tts = SuperhumanTTS(ckpt="bank_large.npz")
    for lang, t in [("en", "Second random round"), ("hi", "दूसरा दौर"),
                    ("bn", "দ্বিতীয় রাউন্ড")]:
        w, s = tts.synth(t, lang)
        cases.append((f"{lang}-hum", np.asarray(w), s, "AI"))
    try:
        x, sr = sf.read("test/input/untitled.wav", always_2d=False)
        cases.append(("USER-FILE", np.asarray(x), sr, "HUMAN"))
    except Exception as e:
        print("user file skip:", str(e)[:60])
    ok = 0
    for name, x, sr, want in cases:
        v = verdict(x, sr, lang=name[:2] if name[:2] in ("hi", "en", "bn") else None)
        good = v["label"] == want
        ok += good
        print(f"{name}: -> {v['label']} (p={v['p_ai']}, m={v.get('model')}) "
              f"{'OK' if good else 'MISS'}", flush=True)
    print(f"RANDOM-V2 {ok}/{len(cases)} = {ok/len(cases):.3f}", flush=True)


if __name__ == "__main__":
    main()
