"""Large evaluation: ~90 cases over every scenario x hi/en/bn."""
import io
import wave
import numpy as np
import soundfile as sf
from piper import PiperVoice
from polyvoice.spoof import verdict, phone_channel, add_noise, add_reverb, speed_perturb

VOICES = {
    "hi": "voices/hi-pratham-medium.onnx",
    "en": "voices/en-lessac-medium.onnx",
    "bn": "voices/bn-google-medium.onnx",
}
LONG_TEXT = {
    "hi": "मेरी ट्रेन छूट गई है, अब मैं क्या करूं? कृपया मेरी मदद करें।",
    "en": "I missed my flight, what should I do now? Please help me.",
    "bn": "আমি আমার ফ্লাইট মিস করেছি, এখন কী করব? দয়া করে সাহায্য করুন।",
}
SHORT_TEXT = {
    "hi": "नमस्ते! आप कैसे हैं?",
    "en": "Hello! How are you?",
    "bn": "নমস্কার! কেমন আছেন?",
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
    # humans: 8/lang + phone copies
    for lang in ["hi", "en", "bn"]:
        for i in [5, 77, 150, 303, 402, 611, 702, 901]:
            try:
                x, sr = sf.read(f"data_voice/{lang}/{lang}_{i:05d}.wav", always_2d=False)
            except Exception:
                continue
            x = np.asarray(x)
            cases.append((f"{lang}-HUMAN", x, sr, "HUMAN"))
            cases.append((f"{lang}-HUMAN-phone", phone_channel(x, sr, seed=i), sr, "HUMAN"))
    # AI long + phone + short per voice
    for lang, m in VOICES.items():
        x, sr = psynth(m, LONG_TEXT[lang])
        cases.append((f"{lang}-AI", x, sr, "AI"))
        cases.append((f"{lang}-AI-phone", phone_channel(x, sr, seed=3), sr, "AI"))
        xs, ss = psynth(m, SHORT_TEXT[lang])
        cases.append((f"{lang}-AI-short", xs, ss, "AI"))
        cases.append((f"{lang}-AI-noisy", add_noise(x, sr, snr_db=12.0, seed=7), sr, "AI"))
    # hums
    tts = SuperhumanTTS(ckpt="bank_large.npz")
    for lang, t in [("en", "Hello test one two"), ("hi", "नमस्ते परीक्षण"),
                    ("bn", "নমস্কার পরীক্ষা")]:
        w, s = tts.synth(t, lang)
        cases.append((f"{lang}-hum", np.asarray(w), s, "AI"))
    # tricky humans: reverb / slowed / clipped
    h, _ = sf.read("data_voice/en/en_00011.wav", always_2d=False)
    cases.append(("en-HUMAN-reverb", add_reverb(np.asarray(h), 16000, 0.6, 6), 16000, "HUMAN"))
    h, _ = sf.read("data_voice/hi/hi_00005.wav", always_2d=False)
    cases.append(("hi-HUMAN-slowed", speed_perturb(np.asarray(h), 16000, 0.85), 16000, "HUMAN"))
    h, _ = sf.read("data_voice/bn/bn_00009.wav", always_2d=False)
    cases.append(("bn-HUMAN-clipped", np.clip(np.asarray(h) * 3.0, -1, 1).astype(np.float32),
                  16000, "HUMAN"))
    # user file
    try:
        x, sr = sf.read("test/input/untitled.wav", always_2d=False)
        cases.append(("USER-FILE", np.asarray(x), sr, "HUMAN"))
    except Exception as e:
        print("user file skip:", str(e)[:60])
    # es/ar AI (known hard, counted separately)
    for lang, m, t in [("es", "voices/es-davefx-medium.onnx", "Hola, cómo estás hoy?"),
                       ("ar", "voices/ar-kareem-medium.onnx", "مرحبا، كيف حالك اليوم؟")]:
        try:
            x, sr = psynth(m, t)
            cases.append((f"{lang}-AI", x, sr, "AI*"))
        except Exception as e:
            print("skip", lang, str(e)[:60])
    ok = tot = 0
    buckets: dict[str, list] = {}
    for name, x, sr, want in cases:
        v = verdict(x, sr)
        if want.endswith("*"):
            print(f"{name}: -> {v['label']} (p_ai={v['p_ai']}) [reference only]", flush=True)
            continue
        good = v["label"] == want
        ok += good
        tot += 1
        buckets.setdefault(name.split("-")[0] + "/" + ("HUMAN" if "HUMAN" in name else "AI"),
                           [0, 0])[good] += 1
        if not good:
            print(f"{name}: -> {v['label']} (p_ai={v['p_ai']}) MISS", flush=True)
    print(f"BIG-EVAL {ok}/{tot} = {ok/tot:.3f}", flush=True)
    for k, (g, b) in sorted(buckets.items()):
        print(f"  {k}: {g} ok / {b} miss", flush=True)


if __name__ == "__main__":
    main()
