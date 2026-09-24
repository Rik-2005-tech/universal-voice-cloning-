"""Fetch FLEURS clips via direct parquet reads (no datasets/torchcodec).

Downloads train parquet for given configs, decodes embedded audio bytes with
soundfile, saves data_voice_fresh/<short>/ (wav + .txt sidecar).
Usage: python3 fetch_fleurs_direct.py --langs cmn_hans_cn,ar_eg --n 400
"""
from __future__ import annotations
import argparse
import io
import os
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) PolyVoiceResearch/1.0"}
SHORT = {"cmn_hans_cn": "zh", "ar_eg": "ar", "fr_fr": "fr", "es_419": "es",
         "hi_in": "hi", "bn_in": "bn", "en_us": "en"}


def fetch(url: str, dst: str) -> None:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as fh:
        while True:
            b = r.read(1024 * 256)
            if not b:
                break
            fh.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="cmn_hans_cn,ar_eg")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="data_voice_fresh")
    a = ap.parse_args()

    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    for cfg in [l.strip() for l in a.langs.split(",") if l.strip()]:
        lang = SHORT.get(cfg, cfg)
        d = os.path.join(a.out, lang)
        os.makedirs(d, exist_ok=True)
        have = len([f for f in os.listdir(d) if f.endswith(".wav")])
        print(f"[{cfg}] have={have} target={a.n}", flush=True)
        if have >= a.n:
            continue
        pq_p = f"/tmp/{cfg}.parquet"
        if not os.path.exists(pq_p):
            rev = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
            url = (f"https://huggingface.co/datasets/google/fleurs/resolve/{rev}/"
                   f"parquet-data/{cfg}/train-00000-of-00001.parquet")
            print(f"[{cfg}] downloading parquet...", flush=True)
            fetch(url, pq_p)
            print(f"[{cfg}] parquet saved", flush=True)
        table = pq.read_table(pq_p)
        cols = table.column_names
        print(f"[{cfg}] rows={table.num_rows} cols={cols}", flush=True)
        tx_col = "transcription" if "transcription" in cols else "raw_transcription"
        n = have
        for i in range(table.num_rows):
            if n >= a.n:
                break
            try:
                au = table.column("audio")[i].as_py()
                raw = au.get("bytes") if isinstance(au, dict) else bytes(au)
                tx = str(table.column(tx_col)[i].as_py() or "").strip()
                if not tx:
                    continue
                x, sr = sf.read(io.BytesIO(raw), always_2d=False)
                x = np.asarray(x, dtype=np.float32)
                if x.ndim == 2:
                    x = x.mean(axis=1)
                if sr != 16000:
                    x = np.interp(np.linspace(0, 1, len(x)),
                                  np.linspace(0, 1, int(round(len(x) / sr * 16000))),
                                  x).astype(np.float32)
                if len(x) < 8000:
                    continue
                name = f"{lang}_{n:05d}"
                sf.write(os.path.join(d, name + ".wav"), x, 16000)
                with open(os.path.join(d, name + ".txt"), "w", encoding="utf-8") as fh:
                    fh.write(tx)
                n += 1
                if n % 100 == 0:
                    print(f"[{cfg}] saved {n}", flush=True)
            except Exception as e:
                continue
        print(f"[{cfg}] DONE saved={n}", flush=True)


if __name__ == "__main__":
    main()
