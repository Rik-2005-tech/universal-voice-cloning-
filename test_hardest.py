"""Hardest-case test: street noise, crowds, shorts x hi/en/bn/fr/es."""
import io
import random
import wave
import numpy as np
import soundfile as sf
from piper import PiperVoice
from polyvoice.spoof import verdict, add_call_noise, mix_voices

rng = random.Random(11)
HUMAN_SRC = {"hi": "data_voice/hi", "en": "data_voice/en", "bn": "data_voice/bn",
             "fr": "data_voice_world/fr", "es": "data_voice_world/es"}
VOICES = {
    "hi": "voices/hi-pratham-medium.onnx",
    "en": "voices/en-lessac-medium.onnx",
    "bn": "voices/bn-google-medium.onnx",
    "fr": "voices/fr-siwis-medium.onnx",
    "es": "voices/es-davefx-medium.onnx",
}
AI_TEXTS = {
    "hi": ["कल बाज़ार में बहुत भीड़ थी।", "यह चाय बहुत गरम है।"],
    "en": ["The market was busy today.", "This tea is very hot."],
    "bn": ["গতকাল বাজারে খুব ভিড় ছিল।", "এই চা খুব গরম।"],
    "fr": ["Le marché était bondé hier.", "Ce thé est très chaud."],
    "es": ["El mercado estaba lleno ayer.", "Este té está muy caliente."],
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


def read_pool(lang):
    import os
    d = HUMAN_SRC[lang]
    files = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
    rng.shuffle(files)
    return [os.path.join(d, f) for f in files[:8]]


def main():
    from polyvoice.backends import SuperhumanTTS
    tts = SuperhumanTTS(ckpt="bank_large.npz")
    ok = tot = 0
    per = {}
    for lang in ["hi", "en", "bn", "fr", "es"]:
        pool = read_pool(lang)
        clips = []
        for f in pool:
            x, sr = sf.read(f, always_2d=False)
            clips.append((np.asarray(x), sr))
        # 1. street-noise humans 5dB
        for x, sr in clips[:4]:
            nz = add_call_noise(x, sr, kind="street", snr_db=5.0, seed=1)
            v = verdict(nz, sr, lang=lang)
            good = v["label"] == "HUMAN"
            ok += good
            tot += 1
            per.setdefault(lang, [0, 0])[good] += 1
            print(f"{lang}-HUMAN-street5: -> {v['label']} p={v['p_ai']} "
                  f"{'OK' if good else 'MISS'}", flush=True)
        # 2. crowd: two humans + street 5dB
        for k in range(3):
            (a, sa), (b, sb) = clips[2 * k], clips[2 * k + 1]
            m = mix_voices(a, b, 16000, voice_snr_db=0.0, seed=k)
            nz = add_call_noise(m, 16000, kind="street", snr_db=5.0, seed=k + 9)
            v = verdict(nz, 16000, lang=lang)
            good = v["label"] == "HUMAN"
            ok += good
            tot += 1
            per.setdefault(lang, [0, 0])[good] += 1
            print(f"{lang}-CROWD5: -> {v['label']} p={v['p_ai']} "
                  f"{'OK' if good else 'MISS'}", flush=True)
        # 3. AI + street noise 8dB
        for t in AI_TEXTS[lang]:
            x, sr = psynth(VOICES[lang], t)
            nz = add_call_noise(x, sr, kind="street", snr_db=8.0, seed=2)
            v = verdict(nz, sr, lang=lang)
            good = v["label"] == "AI"
            ok += good
            tot += 1
            per.setdefault(lang, [0, 0])[good] += 1
            print(f"{lang}-AI-street8: -> {v['label']} p={v['p_ai']} "
                  f"{'OK' if good else 'MISS'}", flush=True)
        # 4. clean AI + clean human spot checks
        x, sr = psynth(VOICES[lang], AI_TEXTS[lang][0])
        for name, w, s, want in [(f"{lang}-AI", x, sr, "AI"),
                                 (f"{lang}-HUMAN", clips[0][0], clips[0][1], "HUMAN")]:
            v = verdict(w, s, lang=lang)
            good = v["label"] == want
            ok += good
            tot += 1
            per.setdefault(lang, [0, 0])[good] += 1
            print(f"{name}: -> {v['label']} p={v['p_ai']} "
                  f"{'OK' if good else 'MISS'}", flush=True)
    print(f"HARDEST {ok}/{tot} = {ok/tot:.3f}", flush=True)
    for lang, (g, b) in sorted(per.items()):
        print(f"  {lang}: {g} ok / {b} miss", flush=True)


if __name__ == "__main__":
    main()
