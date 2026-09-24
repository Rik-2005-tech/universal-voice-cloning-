"""Fetch LibriVox audiobook chapters (fr/es/zh/ar) -> clipped VAD segments.

Uses archive.org metadata (reliable host). No transcripts needed for the
detector (audio + label only). Quality filter applied at save time.
Output: data_voice_world/<lang>/*.wav (16kHz mono).
"""
from __future__ import annotations
import argparse
import os
import shutil
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) PolyVoiceResearch/1.0"}

BOOKS = {
    # clean solo-read audiobooks, one identifier each (more added on demand)
    "fr": ["trois_mousquetaires_0810_librivox",
           "fables_de_la_fontaine_06_jl_librivox"],
    "es": ["tiemposdificiles_2411_librivox",
           "espectros_2011_librivox"],
    "zh": ["art_of_war_chinese_1506_librivox"],
    "ar": ["kalilawadimna_2608_librivox",
           "kitab_adab_dunya_1107_librivox"],
}


def fetch(url: str, timeout=60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def metadata(ident: str) -> dict:
    import json
    return json.loads(fetch(f"https://archive.org/metadata/{ident}").decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_voice_world")
    ap.add_argument("--per-lang", type=int, default=250)
    ap.add_argument("--langs", default="fr,es,zh,ar")
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from polyvoice.audio_io import to_mono_16k, vad_energy
    from polyvoice.quality import is_clean

    for lang in [l.strip() for l in a.langs.split(",") if l.strip()]:
        d = os.path.join(a.out, lang)
        os.makedirs(d, exist_ok=True)
        have = len([f for f in os.listdir(d) if f.endswith(".wav")])
        print(f"[{lang}] have={have} target={a.per_lang}", flush=True)
        if have >= a.per_lang:
            continue
        n = have
        for ident in BOOKS.get(lang, []):
            if n >= a.per_lang:
                break
            try:
                meta = metadata(ident)
            except Exception as e:
                print(f"[{lang}] meta fail {ident}: {str(e)[:80]}", flush=True)
                continue
            mp3s = [f for f in meta.get("files", [])
                    if f.get("name", "").endswith(".mp3") and not f.get("private")]
            print(f"[{lang}] {ident}: {len(mp3s)} mp3s", flush=True)
            for f in mp3s:
                if n >= a.per_lang:
                    break
                url = f"https://archive.org/download/{ident}/{urllib.parse.quote(f['name'])}"
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                        req = urllib.request.Request(url, headers=UA)
                        with urllib.request.urlopen(req, timeout=120) as r, open(tmp.name, "wb") as fh:
                            shutil.copyfileobj(r, fh, length=1024 * 256)
                        x, sr = sf.read(tmp.name, always_2d=False)
                except Exception as e:
                    print(f"[{lang}] dl fail: {str(e)[:70]}", flush=True)
                    continue
                x = to_mono_16k(np.asarray(x, dtype=np.float32), sr, 16000)
                for (s0, s1) in vad_energy(x, 16000):
                    if n >= a.per_lang:
                        break
                    seg = x[s0:s1]
                    dur = len(seg) / 16000
                    if not (1.5 <= dur <= 12.0):
                        continue
                    if not is_clean(seg, 16000):
                        continue
                    name = f"{lang}_{n:05d}"
                    sf.write(os.path.join(d, name + ".wav"), seg, 16000)
                    n += 1
                print(f"[{lang}] saved {n}/{a.per_lang}", flush=True)
        print(f"[{lang}] DONE saved={n}", flush=True)


if __name__ == "__main__":
    main()
