"""One-off cleaner: quarantine junk clips out of data_voice/.

Moves (never deletes) rejected wavs + sidecars into data_voice_rejected/
preserving language subfolders. Run: python3 clean_voice_data.py [--apply]
Default is dry-run (report only).
"""
from __future__ import annotations
import argparse
import os
import shutil
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data_voice")
    ap.add_argument("--dst", default="data_voice_rejected")
    ap.add_argument("--apply", action="store_true", help="move files (default: report only)")
    a = ap.parse_args()

    from polyvoice.quality import quality_report
    import numpy as np

    total, bad = 0, 0
    for lang in sorted(os.listdir(a.src)):
        d = os.path.join(a.src, lang)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".wav"):
                continue
            total += 1
            p = os.path.join(d, f)
            try:
                x, sr = sf.read(p, always_2d=False)
                rep = quality_report(np.asarray(x), sr)
            except Exception as e:
                rep = {"ok": False, "reasons": [f"unreadable:{str(e)[:40]}"]}
            if not rep.get("ok"):
                bad += 1
                print(f"REJECT {lang}/{f}: {rep.get('reasons')}", flush=True)
                if a.apply:
                    dd = os.path.join(a.dst, lang)
                    os.makedirs(dd, exist_ok=True)
                    shutil.move(p, os.path.join(dd, f))
                    side = os.path.splitext(p)[0] + ".txt"
                    if os.path.exists(side):
                        shutil.move(side, os.path.join(dd, os.path.basename(side)))
    print(f"{bad}/{total} rejected ({'moved' if a.apply else 'dry-run'})", flush=True)


if __name__ == "__main__":
    main()
