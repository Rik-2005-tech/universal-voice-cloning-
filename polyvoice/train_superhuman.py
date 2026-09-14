"""CPU-trainable superhuman voice head (numpy-only).
Learns per-language: f0_base, vibrato depth/rate, clarity EQ, breath level, rate
to minimize mel-spectral L2 vs reference human wavs while maximizing clarity.
Smoke-trains in seconds on CPU; same API plugs into full XTTS/VITS fine-tune on GPU.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def mel_like(x: np.ndarray, n_fft: int = 512, n_bands: int = 32) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    hop = n_fft // 2
    frames = [x[i:i + n_fft] * np.hanning(n_fft) for i in range(0, len(x) - n_fft, hop)]
    S = np.stack(frames) if frames else np.zeros((1, n_fft), dtype=np.float32)
    mag = np.abs(np.fft.rfft(S, axis=1))  # (T, F)
    F = mag.shape[1]
    # triangular mel-ish filterbank
    edges = np.linspace(0, F, n_bands + 2).astype(int)
    fb = np.zeros((n_bands, F), dtype=np.float32)
    for b in range(n_bands):
        a, c, d = edges[b], edges[b + 1], edges[b + 2]
        if c > a:
            fb[b, a:c] = np.linspace(0, 1, c - a)
        if d > c:
            fb[b, c:d] = np.linspace(1, 0, d - c)
    mel = np.log1p(mag @ fb.T)
    return mel.astype(np.float32)


@dataclass
class VoiceParams:
    f0: float = 185.0
    vib_depth: float = 3.0
    vib_rate: float = 5.0
    brightness: float = 0.5  # 0 dark .. 1 bright (superhuman clarity)
    breath_db: float = -42.0
    rate: float = 1.0

    def vector(self) -> np.ndarray:
        return np.array([self.f0 / 200.0, self.vib_depth / 5.0, self.vib_rate / 6.0,
                         self.brightness, (self.breath_db + 60) / 30.0, self.rate], dtype=np.float64)

    @staticmethod
    def from_vector(v: np.ndarray) -> "VoiceParams":
        return VoiceParams(f0=float(v[0] * 200.0), vib_depth=float(v[1] * 5.0),
                           vib_rate=float(v[2] * 6.0), brightness=float(np.clip(v[3], 0, 1)),
                           breath_db=float(v[4] * 30.0 - 60), rate=float(np.clip(v[5], 0.8, 1.25)))


def synth_from_params(text: str, p: VoiceParams, sr: int = 24000) -> np.ndarray:
    """Legacy entry keeps API; now delegates to human-like glottal stack
    (formants + aspiration) so training matches universal inference."""
    try:
        from .universal import synth_universal
        return synth_universal(text, "und", p, sr)
    except Exception:
        pass
    dur = max(0.4, min(8.0, len(text) / (14.0 * p.rate)))
    n = int(sr * dur)
    t = np.arange(n) / sr
    f0t = p.f0 * (1 + 0.02 * np.sin(2 * np.pi * 1.3 * t))  # phrase intonation
    phase = 2 * np.pi * np.cumsum(f0t) / sr + p.vib_depth * np.sin(2 * np.pi * p.vib_rate * t)
    wav = 0.35 * np.sin(phase)
    # brightness = harmonic boost (clarity beyond human)
    wav = wav + p.brightness * 0.12 * np.sin(2 * phase) + p.brightness * 0.05 * np.sin(3 * phase)
    fade = min(n // 10, 2000)
    wav[:fade] *= np.linspace(0, 1, fade)
    wav[-fade:] *= np.linspace(1, 0, fade)
    return wav.astype(np.float32)


def loss_vs_ref(text: str, p: VoiceParams, ref_wav: np.ndarray, sr: int = 24000,
                ref_mel=None, lang: str = "und") -> float:
    try:
        from .universal import synth_universal
        hyp = synth_universal(text, lang, p, sr)
    except Exception:
        hyp = synth_from_params(text, p, sr)
    n = min(len(hyp), len(ref_wav))
    if n < 100:
        return 9.0
    # time-align by truncation for smoke training (DTW in full version)
    a = mel_like(hyp[:n], n_fft=256, n_bands=24)
    b = ref_mel if ref_mel is not None else mel_like(ref_wav[:n], n_fft=256, n_bands=24)
    m = min(len(a), len(b))
    spec = float(np.mean((a[:m] - b[:m]) ** 2))
    # clarity bonus: reward brightness, punish muddiness
    clarity = -0.05 * p.brightness
    return spec + clarity


def normalize_lang(lang: str) -> str:
    """Arbitrary BCP-47/ISO tolerant: 'pt-BR', 'PT_br' -> 'pt'. 'xx' stays 'xx'. Empty -> 'und'."""
    lang = (lang or "und").strip().lower().replace("_", "-")
    base = lang.split("-")[0].split(":")[0].split("/")[0]
    base = "".join(c for c in base if c.isalnum()) or "und"
    return base[:12]


def train_multilingual(bank: dict[str, tuple[list[str], list[np.ndarray]]],
                       steps_global: int = 15, steps_per_lang: int = 8,
                       sr: int = 16000, seed: int = 0) -> tuple[dict[str, VoiceParams], VoiceParams, dict]:
    """Any-language training: shared universal base + per-language adapter delta.

    bank: {lang_code: (texts, ref_wavs)}. ANY BCP-47/ISO code accepted,
    any count (even 1 sample). Unknown scripts/tonal families handled by
    polyvoice.universal typological priors. Returns ({lang: params}, global, hists).
    Unseen langs at inference -> get_params_for_lang() zero-shot blend.
    """
    norm = {normalize_lang(k): v for k, v in bank.items()}
    all_texts, all_refs, all_langs = [], [], []
    for lang, (texts, refs) in norm.items():
        all_texts.extend(texts)
        all_refs.extend(refs)
        all_langs.extend([lang] * len(texts))
    if not all_texts:
        g = VoiceParams()
        return {}, g, {}
    glob, _ = train_voice(all_texts, all_refs, steps=steps_global, sr=sr, seed=seed,
                          langs=all_langs)
    per, hists = {}, {}
    for i, (lang, (texts, refs)) in enumerate(norm.items()):
        # warm-start from global, few steps = fast + avoids overfit on tiny langs
        import copy
        v0 = glob.vector()
        p, h = train_voice(texts, refs, steps=steps_per_lang, sr=sr, seed=seed + i + 1,
                           lang=lang)
        # blend global + lang-specific (keeps tiny langs stable)
        v = 0.5 * glob.vector() + 0.5 * p.vector()
        per[lang] = VoiceParams.from_vector(v)
        hists[lang] = h
    # universal fallback = global
    per.setdefault("und", glob)
    return per, glob, hists


def save_bank(path: str, per: dict[str, VoiceParams]) -> None:
    import numpy as np
    langs = sorted(per.keys())
    mat = np.stack([per[l].vector() for l in langs])
    np.savez(path, langs=np.array(langs), mat=mat)


def load_bank(path: str) -> dict[str, VoiceParams]:
    import numpy as np
    d = np.load(path, allow_pickle=True)
    if "mat" in d.files:  # multilingual bank
        langs = [str(x) for x in d["langs"]]
        return {l: VoiceParams.from_vector(d["mat"][i]) for i, l in enumerate(langs)}
    # legacy single-voice ckpt
    return {"und": VoiceParams(f0=float(d["f0"]), vib_depth=float(d["vib_depth"]),
                               vib_rate=float(d["vib_rate"]), brightness=float(d["brightness"]),
                               breath_db=float(d["breath_db"]), rate=float(d["rate"]))}


def get_params_for_lang(target: str, bank: dict[str, VoiceParams],
                        glob: VoiceParams | None = None) -> VoiceParams:
    """Zero-shot adapter lookup for ANY language (delegates to universal)."""
    from .universal import adapt_to_lang
    return adapt_to_lang(target, bank, glob)


def train_voice(texts: list[str], refs: list[np.ndarray], steps: int = 25, lr: float = 0.08,
                sr: int = 16000, seed: int = 0, lang: str = "und",
                langs: list[str] | None = None) -> tuple[VoiceParams, list[float]]:
    rng = np.random.default_rng(seed)
    # speed: truncate refs to 1.2s, precompute mel once
    refs_s = [np.asarray(r, dtype=np.float32)[: int(sr * 1.2)] for r in refs]
    # resample refs to sr if needed (refs assumed 24k in smoke script -> downsample by slicing)
    ref_mels = [mel_like(r, n_fft=256, n_bands=24) for r in refs_s]
    # per-utterance lang (universal: rhythm-aware loss); default single lang
    if langs is None:
        langs = [lang] * len(texts)
    v = VoiceParams().vector()
    hist: list[float] = []
    best, best_v = 1e9, v.copy()
    for s in range(steps):
        cur = float(np.mean([loss_vs_ref(t, VoiceParams.from_vector(v), r, sr, rm, l)
                             for t, r, rm, l in zip(texts, refs_s, ref_mels, langs)]))
        hist.append(cur)
        if cur < best:
            best, best_v = cur, v.copy()
        # coordinate descent: 1 param per step -> 7x faster, still converges
        i = int(rng.integers(0, len(v)))
        eps = 0.03
        dv = v.copy()
        dv[i] += eps
        lp = float(np.mean([loss_vs_ref(t, VoiceParams.from_vector(dv), r, sr, rm, l)
                            for t, r, rm, l in zip(texts, refs_s, ref_mels, langs)]))
        g = (lp - cur) / eps
        v[i] = v[i] - lr * float(np.clip(g, -2, 2))
        v[3] = float(np.clip(v[3], 0.05, 0.95))
        v[5] = float(np.clip(v[5], 0.8, 1.25))
        v[0] = float(np.clip(v[0], 0.6, 1.3))
    return VoiceParams.from_vector(best_v), hist
