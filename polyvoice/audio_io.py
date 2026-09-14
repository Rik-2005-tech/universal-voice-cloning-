"""Audio I/O, enhancement (superhuman clarity), and VAD. Numpy-only."""
from __future__ import annotations
import numpy as np


def to_mono_16k(x: np.ndarray, sr_in: int, sr_out: int = 16000) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if sr_in == sr_out:
        return x
    # linear resample (good enough for prototype; replace with soxr in prod)
    dur = len(x) / float(sr_in)
    n_out = max(1, int(round(dur * sr_out)))
    old_idx = np.linspace(0, 1, len(x))
    new_idx = np.linspace(0, 1, n_out)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


def normalize_loudness(x: np.ndarray, target_rms: float = 0.12, peak: float = 0.89) -> np.ndarray:
    """Superhuman consistency: fixed RMS + peak limit, removes human volume wobble."""
    x = np.asarray(x, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    if rms < 1e-6:
        return x
    x = x * (target_rms / rms)
    m = float(np.max(np.abs(x)) + 1e-12)
    if m > peak:
        x = x * (peak / m)
    return x.astype(np.float32)


def denoise_spectral_gate(x: np.ndarray, sr: int = 16000, thresh_db: float = -28.0) -> np.ndarray:
    """Lightweight noise gate: estimates noise from first 200ms, suppresses it.
    Gives cleaner-than-human input; no dependency needed.
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) < sr // 2:
        return x
    n_fft = 512
    hop = 256
    # frame
    frames = [x[i:i + n_fft] * np.hanning(n_fft) for i in range(0, len(x) - n_fft, hop)]
    if not frames:
        return x
    S = np.stack(frames)
    mag = np.abs(np.fft.rfft(S, axis=1))
    noise = np.median(mag[: max(1, int(0.2 * sr / hop))], axis=0, keepdims=True)
    thresh = noise * (10.0 ** (thresh_db / -20.0) / 10.0)
    # soft mask
    mask = np.clip((mag - thresh) / (thresh + 1e-8), 0.05, 1.0)
    phase = np.angle(np.fft.rfft(S, axis=1))
    Y = mag * mask * np.exp(1j * phase)
    rec = np.fft.irfft(Y, n=n_fft, axis=1)
    out = np.zeros(len(x), dtype=np.float64)
    cnt = np.zeros(len(x), dtype=np.float64)
    for i, fr in enumerate(rec.real):
        s = i * hop
        out[s:s + n_fft] += fr * np.hanning(n_fft)
        cnt[s:s + n_fft] += np.hanning(n_fft) ** 2 + 1e-8
    out = out / np.maximum(cnt, 1e-8)
    return out.astype(np.float32)


def vad_energy(
    x: np.ndarray, sr: int = 16000, frame_ms: int = 30, thresh: float = 0.015, min_speech_ms: int = 200
) -> list[tuple[int, int]]:
    """Energy VAD -> list of (start, end) sample indices. Swap with Silero/webrtcvad in prod."""
    x = np.asarray(x, dtype=np.float32)
    fl = max(1, int(sr * frame_ms / 1000))
    n = len(x) // fl
    if n == 0:
        return []
    e = np.array([float(np.sqrt(np.mean(x[i * fl:(i + 1) * fl] ** 2))) for i in range(n)])
    speech = e > thresh
    segs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if speech[i]:
            j = i
            while j < n and speech[j]:
                j += 1
            if (j - i) * frame_ms >= min_speech_ms:
                segs.append((i * fl, min(len(x), j * fl)))
            i = j
        else:
            i += 1
    return segs
