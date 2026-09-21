"""Voice-data quality filter: reject clipped / music-backed / junk clips.

Radio jingles, music beds and clipped recordings pollute the HUMAN class
(real case: hi_980, a clipped Mobile-Vaani jingle scoring p_ai=0.964).
Used at fetch time (fetch_voice_data) and for one-off cleaning
(clean_voice_data.py). Numpy-only. Never raises.
"""
from __future__ import annotations
import numpy as np


def quality_report(wav, sr: int = 16000) -> dict:
    """Returns {ok, reasons[], clipped_ratio, music_score, ...}."""
    rep: dict = {"ok": True, "reasons": []}
    try:
        x = np.asarray(wav, dtype=np.float64).ravel()
        if len(x) < 1600:
            return {"ok": False, "reasons": ["too-short"]}
        # 1. clipping: samples slammed to full scale (broadcast compression)
        clipped = float(np.mean(np.abs(x) > 0.98))
        rep["clipped_ratio"] = round(clipped, 4)
        if clipped > 0.005:
            rep["ok"] = False
            rep["reasons"].append(f"clipped({clipped:.3f})")
        # 2. music bed: sustained tonal energy + low variation.
        # frame-level: tonal peaks that barely move + steady loudness.
        n_fft, hop = 1024, 512
        frames = [x[i:i + n_fft] * np.hanning(min(n_fft, len(x) - i))
                  for i in range(0, max(1, len(x) - n_fft), hop)]
        S = np.stack(frames) if frames else np.zeros((1, n_fft))
        mag = np.abs(np.fft.rfft(S, axis=1)) + 1e-9
        # tonal persistence: peak-bin index stability across frames
        peaks = np.argmax(mag, axis=1)
        persist = float(np.mean(peaks[1:] == peaks[:-1])) if len(peaks) > 1 else 0.0
        rep["tonal_persist"] = round(persist, 3)
        # loudness steadiness (music beds are compressed flat)
        lrms = 20 * np.log10(np.sqrt((mag ** 2).mean(axis=1)) + 1e-9)
        stead = float(np.percentile(lrms, 90) - np.percentile(lrms, 10))
        rep["loud_range_db"] = round(stead, 1)
        music = 0.6 * persist + 0.4 * max(0.0, 1.0 - stead / 30.0)
        rep["music_score"] = round(float(music), 3)
        if persist > 0.55 and stead < 12.0:
            rep["ok"] = False
            rep["reasons"].append(f"music-bed(persist={persist:.2f},range={stead:.1f}dB)")
        # 3. digital silence / dead air
        rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        rep["rms"] = round(rms, 4)
        if rms < 1e-4:
            rep["ok"] = False
            rep["reasons"].append("silence")
        return rep
    except Exception as e:
        return {"ok": False, "reasons": [f"error:{str(e)[:40]}"]}


def is_clean(wav, sr: int = 16000) -> bool:
    return bool(quality_report(wav, sr).get("ok", False))
