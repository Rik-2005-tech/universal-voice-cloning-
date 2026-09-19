"""Human-indistinguishability layer: adds sub-perceptual human artifacts.
Keeps perfect diction but adds breath, micro-jitter, pauses, tilt.
Target: spectral flatness ~0.12-0.17 (human refs ~0.14) so spoof gap <0.1.
"""
from __future__ import annotations
import numpy as np


def add_breath(wav: np.ndarray, sr: int, level_db: float = -42.0, at_start: bool = True) -> np.ndarray:
    """Inhale at start + faint continuous aspiration (does not whiten spectrum)."""
    wav = np.asarray(wav, dtype=np.float32)
    rng = np.random.default_rng(7)
    # inhale (band-shaped noise: lowpass white via cumulative smoothing)
    n_b = int(sr * 0.12)
    breath = rng.standard_normal(n_b).astype(np.float32)
    # soft lowpass: 2-tap moving average -> breath-like, not white
    breath = (breath + np.concatenate([[0], breath[:-1]])) * 0.5
    breath = breath * (10.0 ** (level_db / 20.0))
    breath *= np.hanning(n_b)
    # continuous bed at -8dB below inhale (very faint, fills spectral valleys)
    bed = rng.standard_normal(len(wav)).astype(np.float32)
    bed = (bed + np.concatenate([[0], bed[:-1]])) * 0.5
    bed *= (10.0 ** ((level_db - 8.0) / 20.0)) * 0.5
    y = wav + bed
    if at_start:
        return np.concatenate([breath, y]).astype(np.float32)
    return np.concatenate([y, breath]).astype(np.float32)


def f0_jitter(wav: np.ndarray, sr: int, f0: float = 185.0, depth: float = 0.006) -> np.ndarray:
    """Sub-perceptual pitch wobble via fractional-delay vibrato.

    Old version re-synthesized a carrier and whitened the spectrum
    (flatness 0.85). This version time-warps the waveform by +-3 samples
    at 4.7Hz + slow random walk: pitch moves, spectrum preserved.
    """
    x = np.asarray(wav, dtype=np.float32)
    n = len(x)
    if n < 32:
        return x
    t = np.arange(n) / sr
    rng = np.random.default_rng(11)
    # slow random walk for natural drift
    wlen = max(1, n // 800)
    walk = np.cumsum(rng.standard_normal(wlen).astype(np.float64)) * 0.3
    walk = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(walk)), walk)
    # delay amplitude for ~depth pitch shift: A = depth*sr/(2*pi*fm)
    fm = 4.7
    A = float(depth * sr / (2 * np.pi * fm))
    A = float(np.clip(A, 0.5, 4.0))
    mod = A * np.sin(2 * np.pi * fm * t + walk)
    idx = np.arange(n, dtype=np.float64) + mod
    idx = np.clip(idx, 0, n - 1)
    y = np.interp(idx, np.arange(n), x).astype(np.float32)
    # mix 90% warped + 10% dry to keep transients/highs, spectrum ~unchanged
    return (0.9 * y + 0.1 * x).astype(np.float32)


def natural_pauses(wav: np.ndarray, sr: int, text: str) -> np.ndarray:
    """Insert 30-60ms silence at commas/clauses like a trained speaker."""
    import re
    ncommas = len(re.findall(r"[,;—–:、।،]", text)) + len(re.findall(r"[.!?؟।。！？]\s", text + " "))
    if ncommas == 0 or len(wav) == 0:
        return wav
    pause = np.zeros(int(sr * 0.045), dtype=np.float32)
    parts = np.array_split(wav, min(ncommas + 1, 4))
    out = parts[0]
    for p in parts[1:]:
        out = np.concatenate([out, pause, p])
    return out.astype(np.float32)


def match_spectral_tilt(wav: np.ndarray, sr: int, tilt: float = 0.10) -> np.ndarray:
    """Gentle human rolloff. Kept subtle (0.10) so flatness stays ~0.14."""
    x = np.asarray(wav, dtype=np.float32)
    if len(x) < 8:
        return x
    S = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    filt = np.exp(-freqs / (4000.0 / max(0.05, tilt * 4))).astype(np.float32)
    low = np.fft.irfft(S * filt, n=len(x)).astype(np.float32)
    return (0.9 * x + 0.1 * low).astype(np.float32)


def humanize(wav: np.ndarray, sr: int, text: str, f0: float = 185.0) -> np.ndarray:
    y = match_spectral_tilt(wav, sr)
    y = f0_jitter(y, sr, f0=f0)
    y = natural_pauses(y, sr, text)
    y = add_breath(y, sr)
    m = float(np.max(np.abs(y)) + 1e-9)
    return (y * min(1.0, 0.89 / m)).astype(np.float32)
