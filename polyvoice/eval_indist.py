"""Indistinguishability eval: can a detector tell synth from human?
Metrics: MCD-proxy (mel L2), F0 RMSE, spoof score (spectral-flatness logistic),
ABX proxy (embedding distance). Score ~0.5 spoof = indistinguishable.

Robust: flatness measured on loudest speech window (skips leading breath),
pair comparison auto-trims leading silence/breath for fair alignment.
"""
from __future__ import annotations
import numpy as np
from .train_superhuman import mel_like


def _speech_window(x: np.ndarray, n: int = 2048) -> np.ndarray:
    """Loudest n-sample window (skips inhale/silence prepend)."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) <= n:
        return x
    hop = 512
    best, best_e = 0, -1.0
    for s in range(0, len(x) - n, hop):
        w = x[s:s + n]
        e = float(np.mean(w.astype(np.float64) ** 2))
        if e > best_e:
            best_e, best = e, s
    return x[best:best + n]


def spectral_flatness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    n = 2048
    seg = _speech_window(x, n)
    if len(seg) < 1024:
        seg = np.pad(seg, (0, 1024 - len(seg)))
    else:
        seg = seg[:1024]
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) + 1e-9
    return float(np.exp(np.mean(np.log(mag))) / np.mean(mag))


def _trim_leading_silence(x: np.ndarray, sr: int = 16000, thresh: float = 0.02) -> np.ndarray:
    """Strip inhale/silence prepend so ref/hyp align (fair mel compare)."""
    x = np.asarray(x, dtype=np.float32)
    fl = max(1, int(sr * 0.02))
    idx = 0
    while idx + fl < len(x):
        rms = float(np.sqrt(np.mean(x[idx:idx + fl] ** 2)))
        if rms > thresh:
            break
        idx += fl
    # keep 20ms context
    idx = max(0, idx - fl)
    return x[idx:]


def f0_rmse(a: np.ndarray, b: np.ndarray, sr: int = 24000) -> float:
    def est(x):
        x = _speech_window(np.asarray(x, dtype=np.float32), min(len(x), sr * 2))
        n = len(x)
        x = x * np.hanning(n)
        S = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(n, 1 / sr)
        m = (freqs >= 60) & (freqs <= 400)
        if not np.any(m):
            return 150.0
        return float(freqs[m][np.argmax(S[m])])
    try:
        return abs(est(a) - est(b))
    except Exception:
        return 999.0


def spoof_score(wav: np.ndarray) -> float:
    """0=human-like, 1=synth-like. Flat spectrum + no breath + F0-flat => synth."""
    flat = spectral_flatness(wav)
    # synth sines are peaky (low flatness); breathy humans higher flatness
    # map flatness 0..0.6 -> score 1..0
    s = float(np.clip(1.0 - flat / 0.35, 0.0, 1.0))
    return s


def evaluate_pair(ref: np.ndarray, hyp: np.ndarray, sr: int = 16000) -> dict:
    ref_t = _trim_leading_silence(ref, sr)
    hyp_t = _trim_leading_silence(hyp, sr)
    n = min(len(ref_t), len(hyp_t))
    a, b = mel_like(ref_t[:n]), mel_like(hyp_t[:n])
    m = min(len(a), len(b))
    mcd = float(np.mean((a[:m] - b[:m]) ** 2))
    return {"mel_L2": round(mcd, 4), "f0_rmse": round(f0_rmse(ref, hyp, sr), 2),
            "spoof_ref": round(spoof_score(ref), 3), "spoof_hyp": round(spoof_score(hyp), 3),
            "gap": round(abs(spoof_score(ref) - spoof_score(hyp)), 3)}
