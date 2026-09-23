"""Unseen-data evaluation: fresh indices never used in training or tests."""
import io
import wave
import numpy as np
import soundfile as sf
from piper import PiperVoice
from polyvoice.spoof import verdict, verdict_auto, phone_channel, add_call_noise, mix_voices

POOL = {"hi": [100, 200, 400, 600, 800, 1000],
        "en": [100, 300, 500, 700, 900, 1000],
        "bn": [100, 200, 400, 600, 800, 1000]}
VOICES = {
    "hi": "voices/hi-pratham-medium.onnx",
    "en": "voices/en-lessac-medium.onnx",
    "bn": "voices/bn-google-medium.onnx",
}
AI_TEXTS = {
    "hi": ["कल बाज़ार में बहुत भीड़ थी।", "यह किताब मैंने कल खरीदी।",
           "बच्चे मैदान में खेल रहे हैं।", "ट्रेन समय पर पहुंचेगी।"],
    "en": ["The market was crowded yesterday.", "I bought this book yesterday.",
           "Children play in the field.", "The train will arrive on time."],
    "bn": ["গতকাল বাজারে খুব ভিড় ছিল।", "এই বইটি আমি গতকাল কিনেছি।",
           "শিশুরা মাঠে খেলছে।", "ট্রেন সময়মতো পৌঁছাবে।"],
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
        got = []
        for i in idxs:
            try:
                x, sr = sf.read(f"data_voice/{lang}/{lang}_{i:05d}.wav", always_2d=False)
                got.append((np.asarray(x), sr))
            except Exception as e:
                print(f"skip {lang}_{i}: {str(e)[:40]}", flush=True)
        for x, sr in got:
            cases.append((f"{lang}-HUMAN", x, sr, "HUMAN"))
            cases.append((f"{lang}-HUMAN-street8",
                          add_call_noise(x, sr, kind="street", snr_db=8.0, seed=11),
                          sr, "HUMAN"))
        # crowd: two humans + street noise
        for k in range(0, len(got) - 1, 2):
            (a, sa), (b, sb) = got[k], got[k + 1]
            m = mix_voices(a, b, 16000, voice_snr_db=0.0, seed=21 + k)
            nz = add_call_noise(m, 16000, kind="street", snr_db=8.0, seed=22 + k)
            cases.append((f"{lang}-CROWD", nz, 16000, "HUMAN"))
    for lang, m in VOICES.items():
        for t in AI_TEXTS[lang]:
            x, sr = psynth(m, t)
            cases.append((f"{lang}-AI", x, sr, "AI"))
            cases.append((f"{lang}-AI-street8",
                          add_call_noise(x, sr, kind="street", snr_db=8.0, seed=31),
                          sr, "AI"))
    tts = SuperhumanTTS(ckpt="bank_large.npz")
    for lang, t in [("en", "Unseen testing round two"), ("hi", "अनदेखा परीक्षण"),
                    ("bn", "অদেখা পরীক্ষা")]:
        w, s = tts.synth(t, lang)
        cases.append((f"{lang}-hum", np.asarray(w), s, "AI"))
    ok = 0
    for name, x, sr, want in cases:
        v = verdict_auto(x, sr) if "-hum" not in name else verdict(x, sr, lang=name[:2])
        good = v["label"] == want
        ok += good
        flag = "OK" if good else "MISS"
        print(f"{name}: -> {v['label']} (p={v.get('p_ai')}/{v.get('p_fake')}, "
              f"m={v.get('model')}) {flag}", flush=True)
    print(f"UNSEEN {ok}/{len(cases)} = {ok/len(cases):.3f}", flush=True)


if __name__ == "__main__":
    main()
