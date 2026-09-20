"""Deep human-vs-AI voice verdict (pretrained wav2vec2 detector, calibrated).

Model: mo-thecreator/Deepfake-audio-detection (labels fake/real).
Calibrated threshold lives in logs/deep_calib.json; override with threshold=.
Lazy-loads transformers+torch on first call. Falls back gracefully.
"""
from __future__ import annotations
import numpy as np

MODEL_ID = "mo-thecreator/Deepfake-audio-detection"
DEFAULT_THRESHOLD = 0.5
_PIPE = None


def _pipe():
    global _PIPE
    if _PIPE is None:
        from transformers import pipeline
        _PIPE = pipeline("audio-classification", model=MODEL_ID, device=-1)
    return _PIPE


def calibrated_threshold(path: str = "logs/deep_calib.json",
                         default: float = DEFAULT_THRESHOLD) -> float:
    try:
        import json
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
        best_t, best_a = default, -1.0
        t = 0.05
        while t < 1.0:
            acc = sum(((r["p"] >= t) == r["y"]) for r in rows) / max(1, len(rows))
            if acc > best_a:
                best_a, best_t = acc, t
            t += 0.05
        return round(best_t, 2)
    except Exception:
        return default


def verdict_deep(wav, sr: int = 16000, threshold: float | None = None) -> dict:
    """Returns {label, p_fake, threshold}. Never raises (UNKNOWN on failure)."""
    try:
        from .audio_io import to_mono_16k
        x = to_mono_16k(np.asarray(wav), int(sr or 16000), 16000).astype(np.float32)
        out = _pipe()({"array": x, "sampling_rate": 16000}, top_k=2)
        s = {o["label"]: float(o["score"]) for o in out}
        p = s.get("fake", 0.0)
        thr = calibrated_threshold() if threshold is None else float(threshold)
        return {"label": "AI" if p >= thr else "HUMAN", "p_fake": round(p, 3),
                "threshold": thr}
    except Exception as e:
        return {"label": "UNKNOWN", "p_fake": -1.0, "error": str(e)[:120]}
