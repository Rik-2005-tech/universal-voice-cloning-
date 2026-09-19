"""Train ONLY hi/en/bn on real voice audio + copy micro voice details.

Stage 1: minibatch VoiceParams (global + per-lang) on data_voice clips.
Stage 2: per language, measure real pitch micro-stats (autocorrelation F0)
         + learn mean spectral residual (real minus our synth) for EQ copy.
Output: bank_micro.npz (params + residuals + jitter stats).
"""
from __future__ import annotations
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_voice")
    ap.add_argument("--langs", default="hi,en,bn")
    ap.add_argument("--per-lang", type=int, default=1200)
    ap.add_argument("--steps-global", type=int, default=150)
    ap.add_argument("--steps-per-lang", type=int, default=25)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--bank-out", default="bank_micro.npz")
    ap.add_argument("--sr", type=int, default=16000)
    a = ap.parse_args()

    from polyvoice.universal import load_training_bank
    from polyvoice.train_superhuman import VoiceParams
    from polyvoice.timbre import f0_stats, learn_residual, save_micro_bank
    from polyvoice.universal import synth_universal
    from polyvoice.humanize import humanize
    from train_large import train_minibatch

    want = [l.strip() for l in a.langs.split(",") if l.strip()]
    raw = load_training_bank(a.data_dir, sr=a.sr)
    rng = np.random.default_rng(7)
    bank: dict[str, tuple[list, list]] = {}
    for lang in want:
        if lang not in raw:
            print(f"WARNING: no '{lang}' in {a.data_dir}, skipping", flush=True)
            continue
        texts, refs = raw[lang]
        k = min(a.per_lang, len(texts))
        idx = sorted(rng.choice(len(texts), size=k, replace=False))
        bank[lang] = ([texts[i] for i in idx], [refs[i] for i in idx])
    print(f"real bank: {[(l, len(v[0])) for l, v in bank.items()]}", flush=True)
    if not bank:
        raise SystemExit("no training data")

    all_texts, all_refs, all_langs = [], [], []
    for lang, (texts, refs) in bank.items():
        all_texts.extend(texts)
        all_refs.extend(refs)
        all_langs.extend([lang] * len(texts))

    glob, gh = train_minibatch(all_texts, all_refs, all_langs,
                               steps=a.steps_global, batch=a.batch,
                               sr=a.sr, seed=0, tag="micro-global")
    print(f"global val {gh[0]:.4f} -> {gh[-1]:.4f}", flush=True)

    per, res, jit = {}, {}, {}
    for li, lang in enumerate(sorted(bank)):
        texts, refs = bank[lang]
        p, h = train_minibatch(texts, refs, [lang] * len(texts),
                               steps=a.steps_per_lang, batch=min(a.batch, 8),
                               sr=a.sr, seed=100 + li, init=glob,
                               tag=f"micro:{lang}")
        v = 0.5 * glob.vector() + 0.5 * p.vector()
        per[lang] = VoiceParams.from_vector(v)
        print(f"[{lang}] {h[0]:.4f} -> {h[-1]:.4f}", flush=True)

        # micro-detail copy on real refs (subset for speed)
        sub = refs[: min(300, len(refs))]
        js = [f0_stats(r, a.sr) for r in sub]
        voiced = [j for j in js if j["voiced"] >= 5]
        mj = float(np.mean([j["jitter_rel"] for j in voiced])) if voiced else 0.0
        mf = float(np.mean([j["mean_f0"] for j in voiced])) if voiced else 0.0
        jit[lang] = {"jitter_rel": round(mj, 4), "mean_f0": round(mf, 1),
                     "n": len(voiced)}
        print(f"[{lang}] real pitch: mean_f0={mf:.1f}Hz jitter={mj:.4f} (n={len(voiced)})",
              flush=True)

        Alpha = per[lang]
        def synth_fn(t, _A=Alpha, _l=lang):
            return humanize(synth_universal(t, _l, _A, sr=a.sr), a.sr, t)
        r = learn_residual(texts, refs, synth_fn, sr=a.sr, max_clips=200, seed=li)
        res[lang] = r
        print(f"[{lang}] residual |.|={float(np.abs(r).mean()):.4f}", flush=True)

    per.setdefault("und", glob)
    res.setdefault("und", np.zeros(24, dtype=np.float32))
    jit.setdefault("und", {"jitter_rel": 0.0, "mean_f0": 0.0, "n": 0})
    save_micro_bank(a.bank_out, per, res, jit, glob)
    print(f"saved {a.bank_out}: langs={sorted(per)} global={glob}", flush=True)


if __name__ == "__main__":
    main()
