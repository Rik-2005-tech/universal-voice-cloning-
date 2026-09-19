"""Micro-detail voice copy: learn real voices' fine texture, apply it at synth.

The 6-knob VoiceParams (pitch/vibrato/brightness/breath/rate) sets the broad
voice. This module copies the *minute* stuff per language:
  1. spectral residual: mean log-mel difference (real refs minus our synth)
     over training clips -> an EQ curve re-applied at synthesis (STFT domain).
  2. pitch micro-stats: real voiced-F0 mean/jitter measured by autocorrelation.
Numpy-only. Never raises (falls back to neutral correction).
"""
from __future__ import annotations
import numpy as np


N_FFT_EQ = 256
HOP_EQ = 64
N_BANDS = 24


def _mel_fb(n_fft: int = N_FFT_EQ, n_bands: int = N_BANDS) -> np.ndarray:
    F = n_fft // 2 + 1
    edges = np.linspace(0, F, n_bands + 2).astype(int)
    fb = np.zeros((n_bands, F), dtype=np.float64)
    for b in range(n_bands):
        a, c, d = edges[b], edges[b + 1], edges[b + 2]
        if c > a:
            fb[b, a:c] = np.linspace(0, 1, c - a)
        if d > c:
            fb[b, c:d] = np.linspace(1, 0, d - c)
    return fb


def _stft(x: np.ndarray, n_fft: int = N_FFT_EQ, hop: int = HOP_EQ):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    win = np.hanning(n_fft)
    frames = [x[i:i + n_fft] * win for i in range(0, len(x) - n_fft + 1, hop)]
    S = np.fft.rfft(np.stack(frames), axis=1)
    return S, win


def _istft(S: np.ndarray, n_fft: int = N_FFT_EQ, hop: int = HOP_EQ, length: int = 0):
    win = np.hanning(n_fft)
    n_frames = S.shape[0]
    n_out = (n_frames - 1) * hop + n_fft
    y = np.zeros(n_out)
    wsum = np.zeros(n_out)
    t = np.fft.irfft(S, n=n_fft, axis=1)
    for i in range(n_frames):
        s = i * hop
        y[s:s + n_fft] += (t[i] * win).real
        wsum[s:s + n_fft] += win ** 2
    y = y / np.maximum(wsum, 1e-8)
    if length:
        y = y[:length]
    return y.astype(np.float32)


def apply_eq(wav: np.ndarray, sr: int, mel_residual, strength: float = 0.6) -> np.ndarray:
    """Reshape synth timbre toward the real voice (mel-residual EQ).

    Residuals are learned in 16kHz-bin space; other rates are converted
    through 16kHz so the correction always lands on the right bands.
    """
    FIT_SR = 16000
    try:
        x = np.asarray(wav, dtype=np.float32)
        if len(x) < 256:
            return x
        orig_sr = int(sr)
        if orig_sr != FIT_SR:
            n16 = max(256, int(round(len(x) / orig_sr * FIT_SR)))
            x = np.interp(np.linspace(0, 1, n16), np.linspace(0, 1, len(x)),
                          x).astype(np.float32)
        r = np.asarray(mel_residual, dtype=np.float64).ravel()
        if r.size != N_BANDS:
            return np.asarray(wav, dtype=np.float32)
        fb = _mel_fb()
        lin_gain, *_ = np.linalg.lstsq(fb, r, rcond=None)  # (F,) log-gains
        lin_gain = np.clip(lin_gain, -1.0, 1.0) * float(strength)
        mult = np.exp(lin_gain)
        S, _ = _stft(x)
        S2 = S * mult[None, :]
        y = _istft(S2, length=len(x))
        m_in = float(np.max(np.abs(x)) + 1e-9)
        m_out = float(np.max(np.abs(y)) + 1e-9)
        y = (y * (m_in / m_out)).astype(np.float32)
        if orig_sr != FIT_SR:
            y = np.interp(np.linspace(0, 1, len(wav)), np.linspace(0, 1, len(y)),
                          y).astype(np.float32)
        return y
    except Exception:
        return np.asarray(wav, dtype=np.float32)


def f0_stats(wav: np.ndarray, sr: int = 16000) -> dict:
    """Voiced pitch mean + relative jitter via autocorrelation. Never raises."""
    try:
        from .audio_io import to_mono_16k
        x = to_mono_16k(np.asarray(wav), sr, 16000).astype(np.float64)
        sr = 16000
        fl = int(sr * 0.03)  # 30ms frames
        f0s = []
        for s in range(0, max(0, len(x) - fl), fl):
            fr = x[s:s + fl] - np.mean(x[s:s + fl])
            rms = float(np.sqrt(np.mean(fr ** 2) + 1e-12))
            if rms < 0.02:
                continue
            ac = np.correlate(fr, fr, mode="full")[len(fr) - 1:]
            ac = ac / (ac[0] + 1e-12)
            lo, hi = int(sr / 400), int(sr / 60)
            if hi >= len(ac):
                continue
            pk = lo + int(np.argmax(ac[lo:hi]))
            if ac[pk] < 0.45:  # unvoiced
                continue
            f0s.append(sr / float(pk))
        f0s = np.array([f for f in f0s if 60 <= f <= 400])
        if len(f0s) < 5:
            return {"mean_f0": 0.0, "jitter_rel": 0.0, "voiced": 0}
        return {"mean_f0": round(float(np.mean(f0s)), 1),
                "jitter_rel": round(float(np.std(f0s) / (np.mean(f0s) + 1e-9)), 4),
                "voiced": int(len(f0s))}
    except Exception:
        return {"mean_f0": 0.0, "jitter_rel": 0.0, "voiced": 0}


def learn_residual(texts, refs, synth_fn, sr: int = 16000,
                   max_clips: int = 200, seed: int = 0) -> np.ndarray:
    """Mean log-mel (real minus synth) over training clips. Zero on failure."""
    try:
        from .train_superhuman import mel_like
        rng = np.random.default_rng(seed)
        n = len(texts)
        idx = rng.choice(n, size=min(max_clips, n), replace=False)
        acc = np.zeros(N_BANDS)
        cnt = 0
        for i in idx:
            r = np.asarray(refs[int(i)], dtype=np.float32)[: int(sr * 1.5)]
            h = np.asarray(synth_fn(texts[int(i)]), dtype=np.float32)[: int(sr * 1.5)]
            m = min(len(r), len(h))
            if m < 800:
                continue
            a = mel_like(r[:m], n_fft=256, n_bands=N_BANDS).mean(axis=0)
            b = mel_like(h[:m], n_fft=256, n_bands=N_BANDS).mean(axis=0)
            acc += (a - b)
            cnt += 1
        if cnt == 0:
            return np.zeros(N_BANDS, dtype=np.float32)
        return (acc / cnt).astype(np.float32)
    except Exception:
        return np.zeros(N_BANDS, dtype=np.float32)


def save_micro_bank(path: str, per: dict, res: dict, jit: dict, glob) -> None:
    langs = sorted(per.keys())
    mat = np.stack([per[l].vector() for l in langs])
    rm = np.stack([np.asarray(res.get(l, np.zeros(N_BANDS)), dtype=np.float32)
                   for l in langs])
    jm = np.array([float(jit.get(l, {}).get("jitter_rel", 0.0)) for l in langs])
    fm = np.array([float(jit.get(l, {}).get("mean_f0", 0.0)) for l in langs])
    np.savez(path, langs=np.array(langs), mat=mat, res=rm, jit=jm, f0m=fm,
             glob=glob.vector())


def load_micro_bank(path: str):
    """Returns (params_dict, aux_dict{res, jit, f0m, glob}). Falls back gracefully."""
    from .train_superhuman import VoiceParams, load_bank
    try:
        d = np.load(path, allow_pickle=True)
        if "res" not in d.files:
            return load_bank(path), {}
        langs = [str(x) for x in d["langs"]]
        per = {l: VoiceParams.from_vector(d["mat"][i]) for i, l in enumerate(langs)}
        aux = {"res": {l: d["res"][i] for i, l in enumerate(langs)},
               "jit": {l: float(d["jit"][i]) for i, l in enumerate(langs)},
               "f0m": {l: float(d["f0m"][i]) for i, l in enumerate(langs)},
               "glob": VoiceParams.from_vector(d["glob"])}
        return per, aux
    except Exception:
        return load_bank(path), {}
