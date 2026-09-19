"""Convert LibriSpeech dev-clean into data_voice/en/ (wav + .txt)."""
import os
import numpy as np
import soundfile as sf

SRC = "/tmp/ls/LibriSpeech/dev-clean"
DST = "data_voice/en"
os.makedirs(DST, exist_ok=True)
N_MAX = 1200
n = 0
for root, _, files in sorted(os.walk(SRC)):
    for f in sorted(files):
        if not f.endswith(".trans.txt"):
            continue
        with open(os.path.join(root, f), encoding="utf-8") as fh:
            for line in fh:
                if n >= N_MAX:
                    break
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue
                utt, tx = parts
                wav_p = os.path.join(root, utt + ".flac")
                if not os.path.exists(wav_p):
                    continue
                try:
                    x, sr = sf.read(wav_p, always_2d=False)
                    x = np.asarray(x, dtype=np.float32)
                    if x.ndim == 2:
                        x = x.mean(axis=1)
                    if sr != 16000:
                        n_out = int(round(len(x) / sr * 16000))
                        x = np.interp(np.linspace(0, 1, n_out),
                                      np.linspace(0, 1, len(x)),
                                      x).astype(np.float32)
                    if len(x) < 8000:
                        continue
                    name = f"en_{n:05d}"
                    sf.write(os.path.join(DST, name + ".wav"), x, 16000)
                    with open(os.path.join(DST, name + ".txt"), "w") as t:
                        t.write(tx.strip().lower())
                    n += 1
                except Exception:
                    continue
            if n >= N_MAX:
                break
    if n >= N_MAX:
        break
print(f"en saved: {n}")
