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
    ap.add_argument("--langs", default="hi,en,bn",
                    help="comma list of human languages to train on (default: hi,en,bn)")
    ap.add_argument("--short-per-voice", type=int, default=15,
                    help="extra SHORT (1-4s) AI clips per voice (fixes short-clip blind spot)")
    ap.add_argument("--harsh", action="store_true",
                    help="extra 5-8dB street-noise copy on AI rows (anti-masking)")
    ap.add_argument("--callnoise", action="store_true",
                    help="extra realistic call-background copy (street/babble/vehicle, 5-12dB)")
    ap.add_argument("--tricky-ai", action="store_true",
                    help="extra tricky AI copies (call-noise/reverb/speed/clip/quiet)")
    ap.add_argument("--crowd-n", type=int, default=0,
                    help="cocktail copies per lang: overlapped voices + street noise (both classes)")
    ap.add_argument("--out", default="detect.npz")
    a = ap.parse_args()

    import os
    import random
    from polyvoice.spoof import (extract_features, train_logreg, save_detector,
                                 predict_proba, phone_channel, augment, FALLBACK_TEXTS,
                                 add_call_noise, add_reverb, speed_perturb, add_noise,
                                 mix_voices, estimate_condition)
    from polyvoice.backends import SuperhumanTTS

    rng = random.Random(0)
    X, y, D, C = [], [], [], []

    def _row(xa, sr, label):
        X.append(extract_features(xa, sr))
        y.append(label)
        D.append(len(xa) / float(sr or 16000))
        try:
            C.append(estimate_condition(xa, sr)["bucket"])
        except Exception:
            C.append("clean")

    def tricky_ai(x, sr, seed):
        """One nasty variant per call, rotating: call-noise / hall reverb /
        tempo shift / clipped-loud / whispered-quiet. All still AI."""
        import numpy as _np
        k = int(seed % 5)
        if k == 0:
            kinds = ["street", "babble", "vehicle"]
            return add_call_noise(x, sr, kind=kinds[int(seed // 5) % 3],
                                  snr_db=float(5 + (seed % 8)), seed=seed + 777)
        if k == 1:
            return add_reverb(x, sr, decay=0.6, seed=seed + 778)
        if k == 2:
            return speed_perturb(x, sr, factor=0.9 if seed % 2 else 1.1)
        if k == 3:
            xc = _np.asarray(x, dtype=_np.float64) * 2.5
            return _np.clip(xc, -1, 1).astype(_np.float32)
        xq = _np.asarray(x, dtype=_np.float64) * 0.15
        xq = xq + _np.random.default_rng(seed + 779).standard_normal(len(xq)) \
            * float(_np.sqrt(_np.mean(xq ** 2) + 1e-12)) / (10.0 ** (12.0 / 20.0))
        return xq.astype(_np.float32)

    def add_row(wav, sr, label, seed, harsh=False, call=False, tricky=False):
        x = np.asarray(wav)
        _row(x, sr, label)
        if a.phone:
            xp = phone_channel(x, sr, seed=seed)
            _row(xp, sr, label)
        if a.augment:
            xa = augment(x, sr, seed=seed + 999)
            _row(xa, sr, label)
        if harsh and label == 1:
            from polyvoice.spoof import add_noise as _nz
            xh = _nz(x, sr, snr_db=float(5 + (seed % 4)), seed=seed + 555)
            _row(xh, sr, label)
        if call:
            xc = add_call_noise(x, sr, snr_db=float(5 + (seed % 8)), seed=seed + 666)
            _row(xc, sr, label)
        if tricky and label == 1:
            _row(tricky_ai(x, sr, seed), sr, label)

    # --- HUMAN: real speech clips ---
    train_langs = [s.strip() for s in (a.langs or "").split(",") if s.strip()] or ["hi", "en", "bn"]

    def _human_dir(lang: str) -> str:
        for cand in (f"data_voice/{lang}", f"data_voice_world/{lang}"):
            if os.path.isdir(cand):
                return cand
        return f"data_voice/{lang}"

    for lang in train_langs:
        d = _human_dir(lang)
        wavs = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
        rng.shuffle(wavs)
        for k, f in enumerate(wavs[:a.human_per_lang]):
            import soundfile as sf
            x, sr = sf.read(os.path.join(d, f), always_2d=False)
            add_row(x, sr, 0, seed=k, call=a.callnoise)
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
                add_row(x, sr, 1, seed=1000 + k, harsh=a.harsh, tricky=a.tricky_ai)
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
                    for s in _re.split(r"(?<=[।?!.。！？؟])\s+", line.strip()):
                        s = s.strip()
                        if 10 <= len(s) <= 60:
                            pool.append(s)
        pool += FALLBACK_TEXTS.get(lang, FALLBACK_TEXTS["en"])
        rng.shuffle(pool)
        n_short = 0
        for k, t in enumerate(pool[:a.short_per_voice]):
            try:
                x, sr = synth_piper(t[:80], model)
                add_row(x, sr, 1, seed=3000 + k, harsh=a.harsh, tricky=a.tricky_ai)
                n_short += 1
            except Exception as e:
                print(f"piper-short skip {lang}: {str(e)[:60]}", flush=True)
        print(f"ai short {lang}: {n_short}", flush=True)
    # --- AI: our own synth hum (focused langs only) ---
    tts = SuperhumanTTS(ckpt="bank_large.npz")
    hum_texts = ["Hello, how are you today?", "नमस्ते, आप कैसे हैं?",
                 "নমস্কার, আপনি কেমন আছেন?", "Bonjour, comment vas-tu?",
                 "Hola, cómo estás hoy?", "你好，你今天怎么样?",
                 "مرحبا، كيف حالك اليوم؟"]
    hum_langs = ["en", "hi", "bn", "fr", "es", "zh", "ar"]
    hum_keep = [i for i in range(a.hum_n) if hum_langs[i % len(hum_langs)] in train_langs]
    for j, i in enumerate(hum_keep):
        lang = hum_langs[i % len(hum_langs)]
        w, sr = tts.synth(hum_texts[i % len(hum_texts)] + f" ({i})", lang)
        add_row(np.asarray(w), sr, 1, seed=2000 + i, harsh=a.harsh, tricky=a.tricky_ai)
        if (j + 1) % 20 == 0:
            print(f"hums: {j + 1}/{len(hum_keep)}", flush=True)
    print(f"ai samples: {sum(y)}", flush=True)
    # --- CROWD: overlapped voices + street noise (noisy streets/crowds) ---
    if a.crowd_n > 0:
        import soundfile as _sf
        pool = {}
        for lang in train_langs:
            d = _human_dir(lang)
            files = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
            rng.shuffle(files)
            pool[lang] = [os.path.join(d, f) for f in files[:max(60, a.crowd_n)]]
        for lang in train_langs:
            made = 0
            for k in range(a.crowd_n):
                try:
                    f1, f2 = rng.sample(pool[lang], 2)
                    x1, s1 = _sf.read(f1, always_2d=False)
                    x2, s2 = _sf.read(f2, always_2d=False)
                    m = mix_voices(np.asarray(x1), np.asarray(x2), 16000,
                                   voice_snr_db=float(rng.uniform(-3, 3)), seed=5000 + k)
                    snr = float(rng.uniform(0, 15))
                    nz = add_call_noise(m, 16000, kind=["street", "babble", "vehicle"][k % 3],
                                        snr_db=snr, seed=6000 + k)
                    add_row(nz, 16000, 0, seed=7000 + k)
                    made += 1
                except Exception as e:
                    print(f"crowd-h skip {lang}: {str(e)[:60]}", flush=True)
            print(f"crowd human {lang}: {made}", flush=True)
        # AI voices talking over human chatter + street noise
        from polyvoice.backends import PiperTTS as _PT
        for lang, model in _PT.VOICE_MAP.items():
            if lang not in train_langs:
                continue
            made = 0
            texts = FALLBACK_TEXTS.get(lang, FALLBACK_TEXTS["en"])
            for k in range(max(1, a.crowd_n // 2)):
                try:
                    x, sr = synth_piper(texts[k % len(texts)], model)
                    f2 = rng.choice(pool[lang])
                    xc, _ = _sf.read(f2, always_2d=False)
                    m = mix_voices(np.asarray(x), np.asarray(xc), sr,
                                   voice_snr_db=float(rng.uniform(0, 6)), seed=8000 + k)
                    snr = float(rng.uniform(0, 15))
                    nz = add_call_noise(m, sr, kind=["street", "babble", "vehicle"][k % 3],
                                        snr_db=snr, seed=9000 + k)
                    add_row(nz, sr, 1, seed=10000 + k)
                    made += 1
                except Exception as e:
                    print(f"crowd-ai skip {lang}: {str(e)[:60]}", flush=True)
            print(f"crowd AI {lang}: {made}", flush=True)

    X = np.stack(X)
    y = np.array(y, dtype=float)
    D = np.array(D, dtype=float)
    C = np.array(C)
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
    thr_map, cond_acc = {}, {}
    dur_b = np.where(D[te] < 4.0, "short", "long")
    for cond in ("clean", "noisy", "crowded"):
        for dur in ("short", "long"):
            m = (C[te] == cond) & (dur_b == dur)
            if m.sum() < 10:
                continue
            bt, ba = 0.5, -1.0
            t = 0.05
            while t < 1.0:
                acc = (((p_te[m] >= t) == y[te][m]).mean())
                if acc > ba:
                    ba, bt = acc, t
                t += 0.05
            thr_map[f"{dur}_{cond}"] = round(bt, 2)
            cond_acc[f"{dur}_{cond}"] = round(ba, 3)
    print(f"condition thresholds: {cond_acc}", flush=True)
    save_detector(a.out, w, b, mu, sd, acc_te, best_s, best_l, thr_map)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
