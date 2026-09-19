"""Convert GramVaani Hindi dev set into data_voice/hi/ (wav + .txt). Resumable."""
import os
import numpy as np
import soundfile as sf

DST = "data_voice/hi"
SRC = "/tmp/dl/GV_Dev_5h"
N_MAX = 1200
os.makedirs(DST, exist_ok=True)
have = {f.split(".")[0] for f in os.listdir(DST) if f.endswith(".wav")}
n = len(have)
for line in open(SRC + "/text", encoding="utf-8"):
    if n >= N_MAX:
        break
    p = line.strip().split(" ", 1)
    if len(p) != 2:
        continue
    utt, tx = p
    if "<" in tx or len(tx.split()) < 3:
        continue
    mp3 = os.path.join(SRC, "Audio", utt + ".mp3")
    if not os.path.exists(mp3):
        continue
    try:
        x, sr = sf.read(mp3, always_2d=False)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if sr != 16000:
            n_out = int(round(len(x) / sr * 16000))
            x = np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, len(x)),
                          x).astype(np.float32)
        if len(x) < 8000:
            continue
        name = f"hi_{n:05d}"
        if name in have:
            n += 1
            continue
        sf.write(os.path.join(DST, name + ".wav"), x, 16000)
        with open(os.path.join(DST, name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(tx.strip())
        have.add(name)
        n += 1
        if n % 100 == 0:
            print(f"hi: {n}", flush=True)
    except Exception:
        continue
print(f"hi DONE: {n}", flush=True)
