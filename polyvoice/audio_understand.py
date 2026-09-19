"""Acoustic understanding: analyse ANY raw audio file with numpy only.

No ASR model needed. Extracts speech features — duration, speech ratio,
loudness, pitch contour (question-like rise?), expressiveness, tempo —
and renders them as a descriptor string the contextual brain understands.
Works for any language since it listens to *how* it sounds, not words.
"""
from __future__ import annotations
import numpy as np


def _to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32)


def estimate_f0_contour(x: np.ndarray, sr: int = 16000, frame_s: float = 0.12) -> list[float]:
    """Per-frame F0 estimate (FFT peak 60-400Hz). Returns Hz per voiced frame."""
    fl = max(256, int(sr * frame_s))
    f0s: list[float] = []
    for s in range(0, max(0, len(x) - fl), fl // 2):
        fr = x[s:s + fl] * np.hanning(fl)
        rms = float(np.sqrt(np.mean(fr ** 2) + 1e-12))
        if rms < 0.01:
            continue
        S = np.abs(np.fft.rfft(fr))
        freqs = np.fft.rfftfreq(fl, 1 / sr)
        m = (freqs >= 60) & (freqs <= 400)
        if not np.any(m):
            continue
        f0s.append(float(freqs[m][np.argmax(S[m])]))
    return f0s


def analyze_audio(x, sr: int = 16000) -> dict:
    """Analyse raw audio -> feature dict + descriptor string. Never raises."""
    try:
        from .audio_io import vad_energy
        x = _to_mono(np.asarray(x, dtype=np.float32))
        dur = len(x) / float(sr or 16000)
        rms = float(np.sqrt(np.mean(x ** 2) + 1e-12)) if len(x) else 0.0
        if rms < 1e-4 or dur < 0.1:
            return {"speech": False, "dur_s": round(dur, 2), "rms": round(rms, 5),
                    "descriptor": "[silence]"}

        segs = vad_energy(x, sr)
        if not segs:
            # audible but no VAD hit (e.g. heavy denoise): treat whole clip as speech
            segs = [(0, len(x))]
        speech_s = sum(b - a for a, b in segs) / float(sr)
        ratio = float(np.clip(speech_s / max(1e-6, dur), 0, 1))

        # energy contour: expressive if it varies a lot
        fl = max(1, int(sr * 0.1))
        nfr = max(1, len(x) // fl)
        env = np.array([float(np.sqrt(np.mean(x[i * fl:(i + 1) * fl] ** 2))) for i in range(nfr)])
        evar = float(np.std(env) / (np.mean(env) + 1e-9))

        # pitch contour: rising end -> question-like; wide range -> expressive
        f0s = estimate_f0_contour(x, sr)
        if len(f0s) >= 4:
            f0_mean = float(np.mean(f0s))
            k = max(1, len(f0s) // 3)
            rise = float(np.mean(f0s[-k:]) - np.mean(f0s[:k]))
            question = bool(rise > 0.04 * f0_mean)
            expressive = bool(evar > 0.55 or (max(f0s) - min(f0s)) > 0.35 * f0_mean)
        elif len(f0s) >= 1:
            f0_mean, question = float(np.mean(f0s)), False
            expressive = bool(evar > 0.55)
        else:
            f0_mean, question, expressive = 0.0, False, bool(evar > 0.55)

        # tempo: voiced-frame rate proxy
        tempo = "fast" if ratio > 0.75 and dur < 4 else ("slow" if ratio < 0.35 else "normal")
        energy = "high" if rms > 0.2 else ("low" if rms < 0.05 else "medium")

        d = {"speech": True, "dur_s": round(dur, 2), "speech_ratio": round(ratio, 2),
             "rms": round(rms, 4), "f0_mean": round(f0_mean, 1),
             "question_like": question, "expressive": expressive,
             "tempo": tempo, "energy": energy, "segments": len(segs)}
        d["descriptor"] = (
            f"[speech dur={d['dur_s']}s segs={len(segs)} "
            f"question={'yes' if question else 'no'} "
            f"energy={energy} tempo={tempo} "
            f"{'expressive' if expressive else 'calm'}]")
        return d
    except Exception as e:
        return {"speech": False, "error": str(e), "descriptor": "[silence]"}
