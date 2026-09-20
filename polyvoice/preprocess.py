"""Input preprocessing: raw user audio -> clean model-ready audio.

Steps: mono -> resample to 16k -> DC removal -> light spectral gate
(conservative, preserves phonetics for language ID) -> loudness normalize
-> leading/trailing silence trim. Returns (clean_audio_16k, info_dict).
Never raises; worst case returns normalized audio with a note.
"""
from __future__ import annotations
import numpy as np


def preprocess_audio(wav, sr_in: int, sr_out: int = 16000,
                     gate_db: float = -20.0, target_rms: float = 0.12,
                     trim_db: float = -40.0) -> tuple[np.ndarray, dict]:
    from .audio_io import to_mono_16k, normalize_loudness, denoise_spectral_gate
    info: dict[str, object] = {"steps": []}
    try:
        x = to_mono_16k(np.asarray(wav), int(sr_in or sr_out), sr_out)
        info["steps"].append(f"mono+resample:{sr_in}->{sr_out}")
        info["dur_in_s"] = round(len(x) / float(sr_out), 2)
        # DC removal (mic offset) — cheap, always safe
        x = (x - float(np.mean(x))).astype(np.float32)
        info["steps"].append("dc-remove")
        # light denoise: conservative gate so fricatives/LID cues survive.
        # Bypass guard: if the gate collapses the signal (>20dB RMS drop),
        # it mistook speech for noise -> keep the un-gated audio.
        try:
            rms_before = float(np.sqrt(np.mean(x ** 2) + 1e-12))
            gated = denoise_spectral_gate(x, sr_out, thresh_db=gate_db, floor=0.15)
            rms_after = float(np.sqrt(np.mean(gated ** 2) + 1e-12))
            drop_db = 20.0 * float(np.log10(rms_before / (rms_after + 1e-12)))
            if drop_db > 20.0:
                info["steps"].append(f"gate-bypassed(collapse {drop_db:.1f}dB)")
            else:
                x = gated
                info["steps"].append(f"spectral-gate({gate_db}dB)")
        except Exception as e:
            info["gate_skipped"] = str(e)
        x = normalize_loudness(x, target_rms=target_rms)
        info["steps"].append(f"loudness-norm(rms={target_rms})")
        # trim digital silence at both ends (keep 50ms context)
        try:
            thr = 10.0 ** (trim_db / 20.0)
            fl = max(1, int(sr_out * 0.02))
            env = np.array([float(np.sqrt(np.mean(x[i:i + fl] ** 2)))
                            for i in range(0, max(1, len(x) - fl), fl)])
            loud = np.where(env > thr)[0]
            if len(loud):
                a = max(0, loud[0] * fl - int(sr_out * 0.05))
                b = min(len(x), (loud[-1] + 1) * fl + int(sr_out * 0.05))
                # safety: never trim away >80% of the clip (threshold miss)
                if (b - a) >= 0.2 * len(x):
                    x = x[a:b]
                    info["steps"].append(f"trim-silence:[{a}:{b}]")
                else:
                    info["steps"].append("trim-skipped(would-cut-too-much)")
        except Exception as e:
            info["trim_skipped"] = str(e)
        info["dur_out_s"] = round(len(x) / float(sr_out), 2)
        info["rms_out"] = round(float(np.sqrt(np.mean(x ** 2) + 1e-12)), 4)
        return x.astype(np.float32), info
    except Exception as e:
        info["error"] = str(e)
        try:
            from .audio_io import to_mono_16k as _m, normalize_loudness as _n
            return _n(_m(np.asarray(wav), int(sr_in or sr_out), sr_out)).astype(np.float32), info
        except Exception:
            return np.zeros(1600, dtype=np.float32), info
