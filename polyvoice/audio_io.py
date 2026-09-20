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


def denoise_spectral_gate(x: np.ndarray, sr: int = 16000, thresh_db: float = -28.0,
                          floor: float = 0.15) -> np.ndarray:
    """Lightweight noise gate. Noise profile comes from the QUIETEST 200ms
    window (first-200ms estimate eats files that start with speech).
    Soft mask with a raised floor so phonetics survive for language ID.
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
    # quietest ~200ms window = true background (never speech onset)
    wframes = max(1, int(0.2 * sr / hop))
    e = (mag ** 2).mean(axis=1)
    if len(e) > wframes:
        s0 = int(np.argmin([e[i:i + wframes].mean()
                            for i in range(0, len(e) - wframes, max(1, wframes // 2))])
                 * max(1, wframes // 2))
        noise = np.median(mag[s0:s0 + wframes], axis=0, keepdims=True)
    else:
        noise = np.median(mag, axis=0, keepdims=True)
    thresh = noise * (10.0 ** (thresh_db / -20.0) / 10.0)
    # soft mask with raised floor (never gate fully shut on speech)
    mask = np.clip((mag - thresh) / (thresh + 1e-8), floor, 1.0)
    phase = np.angle(np.fft.rfft(S, axis=1))
    Y = mag * mask * np.exp(1j * phase)
    # overlap-add WITHOUT re-windowing (frames already windowed at analysis);
    # normalize by overlapped window sum (==1 for hann/50% except edges)
    rec = np.fft.irfft(Y, n=n_fft, axis=1)
    win = np.hanning(n_fft)
    out = np.zeros(len(x), dtype=np.float64)
    wsum = np.zeros(len(x), dtype=np.float64)
    for i, fr in enumerate(rec.real):
        s = i * hop
        out[s:s + n_fft] += fr
        wsum[s:s + n_fft] += win
    out = out / np.maximum(wsum, 1e-3)
    out = out * np.clip(wsum, 0.0, 1.0)  # taper partially-covered edges
    return np.clip(out, -1.0, 1.0).astype(np.float32)


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
