"""Random-data evaluation: fresh indices/texts, never used in training or tests."""
import io
import random
import wave
import numpy as np
import soundfile as sf
from piper import PiperVoice
from polyvoice.spoof import verdict, phone_channel

rng = random.Random(7)
VOICES = {
    "hi": "voices/hi-pratham-medium.onnx",
    "en": "voices/en-lessac-medium.onnx",
    "bn": "voices/bn-google-medium.onnx",
}
AI_TEXTS = {
    "hi": ["मेरी ट्रेन छूट गई है।", "कृपया धीरे बोलें।", "आज बहुत गर्मी है।",
           "बाज़ार कितनी दूर है?", "मुझे एक टैक्सी चाहिए।"],
    "en": ["Please speak slowly.", "It is very hot today.", "How far is the market?",
           "I need a taxi now.", "See you tomorrow morning."],
    "bn": ["দয়া করে ধীরে বলুন।", "আজ খুব গরম।", "বাজার কত দূরে?",
           "আমার এখন ট্যাক্সি দরকার।", "কাল সকালে দেখা হবে।"],
}
# indices never used before (train used shuffled subsets; tests used small sets)
POOL = {"hi": [42, 517, 733, 1105, 208], "en": [64, 388, 540, 826, 1071],
        "bn": [23, 340, 555, 890, 1044]}


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
                print(f"skip {lang}_{i} ({str(e)[:40]})", flush=True)
                continue
            x = np.asarray(x)
            cases.append((f"{lang}-HUMAN", x, sr, "HUMAN"))
            cases.append((f"{lang}-HUMAN-phone", phone_channel(x, sr, seed=i), sr, "HUMAN"))
    for lang, m in VOICES.items():
        for t in AI_TEXTS[lang]:
            x, sr = psynth(m, t)
            cases.append((f"{lang}-AI", x, sr, "AI"))
    tts = SuperhumanTTS(ckpt="bank_large.npz")
    for lang, t in [("en", "Random evaluation song"), ("hi", "अचानक परीक्षण"),
                    ("bn", "হঠাৎ পরীক্ষা")]:
        w, s = tts.synth(t, lang)
        cases.append((f"{lang}-hum", np.asarray(w), s, "AI"))
    try:
        x, sr = sf.read("test/input/untitled.wav", always_2d=False)
        cases.append(("USER-FILE", np.asarray(x), sr, "HUMAN"))
    except Exception as e:
        print("user file skip:", str(e)[:60])
    ok = 0
    for name, x, sr, want in cases:
        v = verdict(x, sr)
        good = v["label"] == want
        ok += good
        flag = "OK" if good else "MISS"
        print(f"{name}: -> {v['label']} (p_ai={v['p_ai']}, thr={v.get('threshold')}) {flag}",
              flush=True)
    print(f"RANDOM {ok}/{len(cases)} = {ok/len(cases):.3f}", flush=True)


if __name__ == "__main__":
    main()
