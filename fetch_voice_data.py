"""Fetch real speech (FLEURS: Hindi/Bengali/English) into data_voice/.

Layout for train_voice.py --data: data_voice/<lang>/*.wav + same-name .txt
Uses streaming so only the first N clips per language are downloaded.
"""
from __future__ import annotations
import argparse
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_voice")
    ap.add_argument("--n", type=int, default=1200, help="clips per language")
    ap.add_argument("--langs", default="hi_in,bn_in,en_us")
    a = ap.parse_args()

    from datasets import load_dataset
    import soundfile as sf

    code_map = {"hi_in": "hi", "bn_in": "bn", "en_us": "en"}
    for cfg in a.langs.split(","):
        lang = code_map.get(cfg, cfg)
        d = os.path.join(a.out, lang)
        os.makedirs(d, exist_ok=True)
        have = len([f for f in os.listdir(d) if f.endswith(".wav")])
        print(f"[{cfg}] have={have} target={a.n}", flush=True)
        if have >= a.n:
            continue
        ds = None
        import time
        for attempt in range(40):
            try:
                ds = load_dataset("google/fleurs", cfg, split="train", streaming=True)
                break
            except Exception as e:
                print(f"[{cfg}] load retry {attempt+1}/40 ({type(e).__name__})", flush=True)
                time.sleep(30)
        if ds is None:
            print(f"[{cfg}] FAILED to load, skipping", flush=True)
            continue
        i = have
        for ex in ds:
            if i >= a.n:
                break
            try:
                au = ex["audio"]
                x = np.asarray(au["array"], dtype=np.float32)
                sr = int(au["sampling_rate"])
                if x.ndim == 2:
                    x = x.mean(axis=1)
                if sr != 16000:
                    n_out = int(round(len(x) / sr * 16000))
                    x = np.interp(np.linspace(0, 1, n_out),
                                  np.linspace(0, 1, len(x)), x).astype(np.float32)
                tx = (ex.get("transcription") or ex.get("raw_transcription") or "").strip()
                if not tx or len(x) < 8000:
                    continue
                m = float(np.sqrt(np.mean(x ** 2) + 1e-12))
                if m < 1e-4:  # skip digital silence
                    continue
                name = f"{lang}_{i:05d}"
                sf.write(os.path.join(d, name + ".wav"), x, 16000)
                with open(os.path.join(d, name + ".txt"), "w", encoding="utf-8") as fh:
                    fh.write(tx)
                i += 1
                if i % 100 == 0:
                    print(f"[{cfg}] saved {i}", flush=True)
            except Exception as e:
                print(f"[{cfg}] skip ({e})", flush=True)
                continue
        print(f"[{cfg}] DONE saved={i}", flush=True)


if __name__ == "__main__":
    main()
