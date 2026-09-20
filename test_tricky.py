"""Tricky audio tests: noisy AI, reverbed human, tiny clips, slow human,
clipped human, crude hum, es/ar AI, user file."""
import io
import wave
import numpy as np
import soundfile as sf
from piper import PiperVoice
from polyvoice.spoof import verdict, phone_channel, add_noise, add_reverb, speed_perturb


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


def load16(path):
    x, sr = sf.read(path, always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != 16000:
        n = int(round(len(x) / sr * 16000))
        x = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x).astype(np.float32)
    return x


def main():
    from polyvoice.backends import SuperhumanTTS
    cases = []
    # 1. AI hidden in street noise
    x, sr = psynth("voices/en-lessac-medium.onnx", "I missed my flight, what should I do now?")
    cases.append(("AI-noisy", add_noise(x, sr, snr_db=8.0, seed=5), sr, "AI"))
    # 2. human in a hall (reverb)
    h, _ = sf.read("data_voice/en/en_00011.wav", always_2d=False)
    cases.append(("HUMAN-reverb", add_reverb(np.asarray(h), 16000, decay=0.6, seed=6), 16000, "HUMAN"))
    # 3. tiny AI fragments
    for lang, m, t in [("en", "voices/en-lessac-medium.onnx", "Hello!"),
                       ("hi", "voices/hi-pratham-medium.onnx", "नमस्ते!")]:
        x, sr = psynth(m, t)
        cases.append((f"{lang}-AI-tiny", x, sr, "AI"))
    # 4. slowed human
    h, _ = sf.read("data_voice/hi/hi_00005.wav", always_2d=False)
    cases.append(("HUMAN-slowed", speed_perturb(np.asarray(h), 16000, 0.85), 16000, "HUMAN"))
    # 5. clipped loud human
    h, _ = sf.read("data_voice/bn/bn_00009.wav", always_2d=False)
    hc = np.clip(np.asarray(h) * 3.0, -1, 1).astype(np.float32)
    cases.append(("HUMAN-clipped", hc, 16000, "HUMAN"))
    # 6. phone-quality AI
    x, sr = psynth("voices/bn-google-medium.onnx", "আমি আমার ফ্লাইট মিস করেছি, এখন কী করব?")
    cases.append(("AI-phone", phone_channel(x, sr, seed=9), sr, "AI"))
    # 7. crude sine hum
    tts = SuperhumanTTS(ckpt="bank_large.npz")
    w, s = tts.synth("Hello test one two three", "en")
    cases.append(("AI-hum", np.asarray(w), s, "AI"))
    # 8/9. es/ar AI (known hard)
    for lang, m, t in [("es", "voices/es-davefx-medium.onnx", "Hola, cómo estás hoy?"),
                       ("ar", "voices/ar-kareem-medium.onnx", "مرحبا، كيف حالك اليوم؟")]:
        x, sr = psynth(m, t)
        cases.append((f"{lang}-AI", x, sr, "AI"))
    # 10. user's own file
    try:
        cases.append(("USER-FILE", load16("test/input/untitled.wav"), 16000, "HUMAN"))
    except Exception as e:
        print("user file skip:", str(e)[:60])
    ok = 0
    for name, x, sr, want in cases:
        v = verdict(x, sr)
        good = v["label"] == want
        ok += good
        print(f"{name}: -> {v['label']} (p_ai={v['p_ai']}, thr={v.get('threshold')}) "
              f"{'OK' if good else 'MISS'}", flush=True)
    print(f"TRICKY {ok}/{len(cases)}", flush=True)


if __name__ == "__main__":
    main()
