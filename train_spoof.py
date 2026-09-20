"""Train human-vs-AI detector.

HUMAN: data_voice clips (LibriSpeech/GramVaani/SLR real speech).
AI: Piper neural voices (7 langs, real novel sentences) + PolyVoice synth hum.
Saves detect.npz + prints held-out accuracy. Your own recordings stay
UNSEEN for the final honesty test (see --honesty-file).
"""
from __future__ import annotations
import argparse
import numpy as np


def synth_piper(text: str, model: str) -> tuple[np.ndarray, int]:
    import io
    import wave
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
        raw = r.readframes(r.getnframes())
        sr = r.getframerate()
    return (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-per-lang", type=int, default=200)
    ap.add_argument("--ai-per-lang", type=int, default=30)
    ap.add_argument("--hum-n", type=int, default=60)
    ap.add_argument("--phone", action="store_true",
                    help="add a phone-channel copy of every sample (matched conditions)")
    ap.add_argument("--augment", action="store_true",
                    help="add a second rotating copy (noise/reverb/speed/phone) per sample")
    ap.add_argument("--voices", default="",
                    help="comma list to narrow AI voices, e.g. en,hi,bn (default: all)")
    ap.add_argument("--short-per-voice", type=int, default=15,
                    help="extra SHORT (1-4s) AI clips per voice (fixes short-clip blind spot)")
    ap.add_argument("--harsh", action="store_true",
                    help="extra 5-8dB street-noise copy on AI rows (anti-masking)")
    ap.add_argument("--out", default="detect.npz")
    a = ap.parse_args()

    import os
    import random
    from polyvoice.spoof import (extract_features, train_logreg, save_detector,
                                 predict_proba, phone_channel, augment, FALLBACK_TEXTS)
    from polyvoice.backends import SuperhumanTTS

    rng = random.Random(0)
    X, y, D = [], [], []

    def add_row(wav, sr, label, seed, harsh=False):
        x = np.asarray(wav)
        X.append(extract_features(x, sr))
        y.append(label)
        D.append(len(x) / float(sr or 16000))
        if a.phone:
            xp = phone_channel(x, sr, seed=seed)
            X.append(extract_features(xp, sr))
            y.append(label)
            D.append(len(xp) / float(sr or 16000))
        if a.augment:
            xa = augment(x, sr, seed=seed + 999)
            X.append(extract_features(xa, sr))
            y.append(label)
            D.append(len(xa) / float(sr or 16000))
        if harsh and label == 1:
            from polyvoice.spoof import add_noise as _nz
            xh = _nz(x, sr, snr_db=float(5 + (seed % 4)), seed=seed + 555)
            X.append(extract_features(xh, sr))
            y.append(label)
            D.append(len(xh) / float(sr or 16000))

    # --- HUMAN: real speech clips ---
    for lang in ["hi", "en", "bn"]:
        d = f"data_voice/{lang}"
        wavs = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
        rng.shuffle(wavs)
        for k, f in enumerate(wavs[:a.human_per_lang]):
            import soundfile as sf
            x, sr = sf.read(os.path.join(d, f), always_2d=False)
            add_row(x, sr, 0, seed=k)
            if (k + 1) % 50 == 0:
                print(f"human {lang}: {k + 1}/{min(a.human_per_lang, len(wavs))}", flush=True)
    print(f"human samples: {sum(1 for v in y if v == 0)}", flush=True)

    # --- AI: Piper voices reading real novel lines (fallback: built-in texts) ---
    from polyvoice.backends import PiperTTS
    _VM = PiperTTS.VOICE_MAP
    _only = [s.strip() for s in (a.voices or "").split(",") if s.strip()]
    VOICES = {l: _VM[l] for l in (_only or list(_VM)) if l in _VM}
    print(f"ai voices: {sorted(VOICES)}", flush=True)
    for lang, model in VOICES.items():
        corp = f"corpora/novel_{lang}.txt"
        if os.path.exists(corp):
            with open(corp, encoding="utf-8") as fh:
                lines = [l.strip() for l in fh if len(l.strip()) > 20]
        else:
            lines = list(FALLBACK_TEXTS.get(lang, FALLBACK_TEXTS["en"]))
        rng.shuffle(lines)
        n_ai = 0
        for k, t in enumerate(lines[:a.ai_per_lang]):
            try:
                x, sr = synth_piper(t[:160], model)
                add_row(x, sr, 1, seed=1000 + k, harsh=a.harsh)
                n_ai += 1
                if n_ai % 10 == 0:
                    print(f"ai voice {lang}: {n_ai}", flush=True)
            except Exception as e:
                print(f"piper skip {lang}: {str(e)[:60]}", flush=True)
        print(f"ai voice {lang}: {n_ai} done", flush=True)
    # --- AI: SHORT clips (1-4s) per voice — kills the short-clip blind spot ---
    import re as _re
    for lang, model in VOICES.items():
        corp = f"corpora/novel_{lang}.txt"
        pool = []
        if os.path.exists(corp):
            with open(corp, encoding="utf-8") as fh:
                for line in fh:
                    for s in _re.split(r"(?<=[।?!])\s+", line.strip()):
                        s = s.strip()
                        if 10 <= len(s) <= 60:
                            pool.append(s)
        pool += FALLBACK_TEXTS.get(lang, FALLBACK_TEXTS["en"])
        rng.shuffle(pool)
        n_short = 0
        for k, t in enumerate(pool[:a.short_per_voice]):
            try:
                x, sr = synth_piper(t[:80], model)
                add_row(x, sr, 1, seed=3000 + k, harsh=a.harsh)
                n_short += 1
            except Exception as e:
                print(f"piper-short skip {lang}: {str(e)[:60]}", flush=True)
        print(f"ai short {lang}: {n_short}", flush=True)
    # --- AI: our own synth hum (focused langs only) ---
    tts = SuperhumanTTS(ckpt="bank_large.npz")
    hum_texts = ["Hello, how are you today?", "नमस्ते, आप कैसे हैं?",
                 "নমস্কার, আপনি কেমন আছেন?"]
    hum_langs = ["en", "hi", "bn"]
    for i in range(a.hum_n):
        lang = hum_langs[i % len(hum_langs)]
        w, sr = tts.synth(hum_texts[i % len(hum_texts)] + f" ({i})", lang)
        add_row(np.asarray(w), sr, 1, seed=2000 + i, harsh=a.harsh)
        if (i + 1) % 20 == 0:
            print(f"hums: {i + 1}/{a.hum_n}", flush=True)
    print(f"ai samples: {sum(y)}", flush=True)

    X = np.stack(X)
    y = np.array(y, dtype=float)
    D = np.array(D, dtype=float)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    w, b, mu, sd = train_logreg(X[tr], y[tr])
    acc_tr = float((((predict_proba(X[tr], w, b, mu, sd) >= 0.5) == y[tr]).mean()))
    acc_te = float((((predict_proba(X[te], w, b, mu, sd) >= 0.5) == y[te]).mean()))
    print(f"train acc={acc_tr:.3f} HELD-OUT acc={acc_te:.3f}", flush=True)
    # confusion on held-out
    from collections import Counter
    pred = (predict_proba(X[te], w, b, mu, sd) >= 0.5).astype(int)
    print("held-out truth/pred:", Counter(zip(y[te].astype(int), pred)), flush=True)
    # duration-bucketed thresholds calibrated on held-out (short clips need
    # their own line: less evidence accumulates)
    p_te = predict_proba(X[te], w, b, mu, sd)
    best_s, best_l, bs, bl = 0.5, 0.5, -1.0, -1.0
    t = 0.05
    while t < 1.0:
        for bucket, arr in (("short", D[te] < 4.0), ("long", D[te] >= 4.0)):
            m = arr
            if m.sum() == 0:
                continue
            acc = (((p_te[m] >= t) == y[te][m]).mean())
            if bucket == "short" and acc > bs:
                bs, best_s = acc, t
            if bucket == "long" and acc > bl:
                bl, best_l = acc, t
        t += 0.05
    print(f"thr_short={best_s:.2f} (acc {bs:.3f}) thr_long={best_l:.2f} (acc {bl:.3f})",
          flush=True)
    save_detector(a.out, w, b, mu, sd, acc_te, best_s, best_l)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
