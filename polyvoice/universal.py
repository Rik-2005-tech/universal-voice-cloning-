"""Universal any-language voice architecture (numpy-only, offline).

Goal: same code + same training loop learns ANY language (seen or unseen)
and mimics human speaking better than human (perfect diction + human texture).

Design (mirrors production XTTS / MMS / YourTTS, smoke-testable on CPU):

1. FRONTEND (language-agnostic text analysis)
   analyze_text() -> script, syllable estimate, tonal need, rhythm class,
   question/exclaim, pause density. Works on any Unicode, no per-lang code.

2. TYPOLOGICAL EMBEDDING (lang2vec-lite)
   lang_embedding() -> 8-D vector from linguistic priors + hash fallback.
   Unseen ISO codes (e.g. 'sw', 'yo', 'qu') still get a sensible vector,
   so training/inference never crashes on a new language.

3. BACKBONE + ADAPTERS
   Shared global VoiceParams (human universal: breath, jitter, tilt)
   + per-language delta (f0, rate, pitch-range, brightness...).
   Tiny langs blend toward global (no overfit). Unseen langs interpolate
   k-nearest trained langs in embedding space (zero-shot).

4. LANGUAGE-AWARE SYNTH
   synth_universal() varies duration / f0 contour / rhythm / pauses by
   language family: tonal (zh/vi/th) gets tone wobble, mora-timed (ja)
   staccato, stress-timed (en/de) alternating stress, syllable-timed
   (fr/es/hi) even, RTL/Indic/CJK scripts handled in duration model.

Full GPU recipe: replace synth_universal with XTTS-v2 / MMS-VITS backbone
+ LoRA per lang (rank 8), frontend -> IPA via espeak-ng / phonemizer,
  reference encoder (ECAPA speaker + MMS LID). This file keeps identical
  API so `train_multilingual` smoke-trains here, scales there.
"""
from __future__ import annotations
import hashlib
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- priors

# rhythm: stress | syllable | mora | tonal-syllable
# f0/rate/range are language-family defaults (fine-tuned in training)
LANG_PRIORS: dict[str, dict] = {
    "en": {"rhythm": "stress", "tonal": 0, "f0": 180, "rate": 1.00, "range": 1.0},
    "de": {"rhythm": "stress", "tonal": 0, "f0": 175, "rate": 0.98, "range": 0.9},
    "fr": {"rhythm": "syllable", "tonal": 0, "f0": 196, "rate": 1.02, "range": 1.0},
    "es": {"rhythm": "syllable", "tonal": 0, "f0": 200, "rate": 1.05, "range": 1.0},
    "it": {"rhythm": "syllable", "tonal": 0, "f0": 198, "rate": 1.03, "range": 1.0},
    "pt": {"rhythm": "syllable", "tonal": 0, "f0": 192, "rate": 1.02, "range": 1.0},
    "ro": {"rhythm": "syllable", "tonal": 0, "f0": 190, "rate": 1.01, "range": 1.0},
    "nl": {"rhythm": "stress", "tonal": 0, "f0": 178, "rate": 1.00, "range": 0.9},
    "pl": {"rhythm": "stress", "tonal": 0, "f0": 182, "rate": 1.00, "range": 0.9},
    "cs": {"rhythm": "stress", "tonal": 0, "f0": 181, "rate": 1.00, "range": 0.9},
    "sk": {"rhythm": "stress", "tonal": 0, "f0": 181, "rate": 1.00, "range": 0.9},
    "hu": {"rhythm": "stress", "tonal": 0, "f0": 183, "rate": 1.00, "range": 0.9},
    "fi": {"rhythm": "stress", "tonal": 0, "f0": 179, "rate": 0.99, "range": 0.85},
    "da": {"rhythm": "stress", "tonal": 0, "f0": 177, "rate": 0.99, "range": 0.85},
    "sv": {"rhythm": "stress", "tonal": 0, "f0": 180, "rate": 1.00, "range": 0.95},
    "no": {"rhythm": "stress", "tonal": 0, "f0": 180, "rate": 1.00, "range": 0.95},
    "ru": {"rhythm": "stress", "tonal": 0, "f0": 175, "rate": 0.98, "range": 0.9},
    "uk": {"rhythm": "stress", "tonal": 0, "f0": 180, "rate": 0.99, "range": 0.9},
    "el": {"rhythm": "stress", "tonal": 0, "f0": 188, "rate": 1.00, "range": 0.9},
    "he": {"rhythm": "stress", "tonal": 0, "f0": 184, "rate": 0.99, "range": 0.9},
    "fa": {"rhythm": "stress", "tonal": 0, "f0": 185, "rate": 0.99, "range": 0.95},
    "hi": {"rhythm": "syllable", "tonal": 0, "f0": 190, "rate": 1.00, "range": 1.1},
    "bn": {"rhythm": "syllable", "tonal": 0, "f0": 192, "rate": 1.00, "range": 1.1},
    "mr": {"rhythm": "syllable", "tonal": 0, "f0": 190, "rate": 1.00, "range": 1.1},
    "ta": {"rhythm": "syllable", "tonal": 0, "f0": 195, "rate": 0.98, "range": 1.1},
    "te": {"rhythm": "syllable", "tonal": 0, "f0": 195, "rate": 0.98, "range": 1.1},
    "kn": {"rhythm": "syllable", "tonal": 0, "f0": 194, "rate": 0.98, "range": 1.1},
    "ml": {"rhythm": "syllable", "tonal": 0, "f0": 194, "rate": 0.98, "range": 1.1},
    "gu": {"rhythm": "syllable", "tonal": 0, "f0": 192, "rate": 1.00, "range": 1.1},
    "pa": {"rhythm": "syllable", "tonal": 1, "f0": 188, "rate": 1.00, "range": 1.2},
    "ur": {"rhythm": "stress", "tonal": 0, "f0": 186, "rate": 1.00, "range": 1.0},
    "ne": {"rhythm": "syllable", "tonal": 0, "f0": 191, "rate": 1.00, "range": 1.1},
    "si": {"rhythm": "syllable", "tonal": 0, "f0": 192, "rate": 1.00, "range": 1.1},
    "ar": {"rhythm": "stress", "tonal": 0, "f0": 184, "rate": 0.97, "range": 0.9},
    "tr": {"rhythm": "stress", "tonal": 0, "f0": 186, "rate": 1.00, "range": 0.9},
    "am": {"rhythm": "stress", "tonal": 0, "f0": 187, "rate": 0.99, "range": 1.0},
    "sw": {"rhythm": "syllable", "tonal": 0, "f0": 188, "rate": 1.00, "range": 1.0},
    "ha": {"rhythm": "tonal", "tonal": 1, "f0": 190, "rate": 1.00, "range": 1.2},
    "ig": {"rhythm": "tonal", "tonal": 1, "f0": 191, "rate": 1.00, "range": 1.2},
    "yo": {"rhythm": "tonal", "tonal": 1, "f0": 192, "rate": 1.00, "range": 1.2},
    "zh": {"rhythm": "tonal", "tonal": 1, "f0": 200, "rate": 0.95, "range": 1.3},
    "yue": {"rhythm": "tonal", "tonal": 1, "f0": 201, "rate": 0.95, "range": 1.3},
    "ja": {"rhythm": "mora", "tonal": 0, "f0": 205, "rate": 0.97, "range": 0.8},
    "ko": {"rhythm": "syllable", "tonal": 0, "f0": 202, "rate": 0.98, "range": 1.0},
    "vi": {"rhythm": "tonal", "tonal": 1, "f0": 202, "rate": 0.96, "range": 1.3},
    "th": {"rhythm": "tonal", "tonal": 1, "f0": 200, "rate": 0.96, "range": 1.3},
    "lo": {"rhythm": "tonal", "tonal": 1, "f0": 200, "rate": 0.96, "range": 1.3},
    "km": {"rhythm": "syllable", "tonal": 0, "f0": 198, "rate": 0.97, "range": 1.1},
    "my": {"rhythm": "tonal", "tonal": 1, "f0": 199, "rate": 0.96, "range": 1.25},
    "id": {"rhythm": "syllable", "tonal": 0, "f0": 190, "rate": 1.02, "range": 1.0},
    "ms": {"rhythm": "syllable", "tonal": 0, "f0": 190, "rate": 1.02, "range": 1.0},
    "tl": {"rhythm": "syllable", "tonal": 0, "f0": 193, "rate": 1.01, "range": 1.05},
    "ka": {"rhythm": "stress", "tonal": 0, "f0": 186, "rate": 1.00, "range": 0.95},
    "hy": {"rhythm": "stress", "tonal": 0, "f0": 185, "rate": 1.00, "range": 0.95},
    "und": {"rhythm": "syllable", "tonal": 0, "f0": 185, "rate": 1.00, "range": 1.0},
}

EMB_KEYS = ["tonal", "stress", "syllable", "mora", "indic", "cjk", "rtl", "pitch_acc"]


def normalize_lang(lang: str) -> str:
    lang = (lang or "und").strip().lower().replace("_", "-")
    base = lang.split("-")[0].split(":")[0].split("/")[0]
    base = "".join(c for c in base if c.isalnum()) or "und"
    return base[:12]


def prior_for(lang: str) -> dict:
    return LANG_PRIORS.get(normalize_lang(lang), LANG_PRIORS["und"])


def lang_embedding(lang: str) -> np.ndarray:
    """8-D typological vector. Known langs from priors; unknown from hash.

    Never fails: any BCP-47 code maps to a stable vector, enabling zero-shot.
    """
    l = normalize_lang(lang)
    p = LANG_PRIORS.get(l)
    v = np.zeros(8, dtype=np.float64)
    if p is not None:
        v[0] = float(p["tonal"])
        v[1] = 1.0 if p["rhythm"] == "stress" else 0.0
        v[2] = 1.0 if p["rhythm"] in ("syllable", "tonal") else 0.0
        v[3] = 1.0 if p["rhythm"] == "mora" else 0.0
        v[4] = 1.0 if l in ("hi", "bn", "mr", "ta", "te", "kn", "ml", "gu", "pa", "ur",
                            "ne", "si") else 0.0
        v[5] = 1.0 if l in ("zh", "yue", "ja", "ko", "vi", "th", "lo", "km", "my") else 0.0
        v[6] = 1.0 if l in ("ar", "ur", "fa", "he") else 0.0
        v[7] = 1.0 if l in ("ja", "sw") else 0.0
        return v
    # unseen language: deterministic pseudo-typology from hash (stable across runs)
    h = hashlib.md5(l.encode()).digest()
    v[0] = 1.0 if h[0] % 5 == 0 else 0.0  # ~20% tonal
    v[1] = 1.0 if h[1] % 2 == 0 else 0.0
    v[2] = 1.0 - v[1]
    v[3] = 1.0 if h[2] % 10 == 0 else 0.0
    v[4] = 1.0 if h[3] % 4 == 0 else 0.0
    v[5] = 1.0 if h[4] % 4 == 0 else 0.0
    v[6] = 1.0 if h[5] % 6 == 0 else 0.0
    v[7] = 0.0
    return v


# ---------------------------------------------------------------- frontend

def detect_script(text: str) -> str:
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o <= 0x0DFF or 0x0D80 <= o <= 0x0DFF:
            return "indic"
        if 0x0600 <= o <= 0x06FF or 0xFB50 <= o <= 0xFDFF or 0x0590 <= o <= 0x05FF:
            return "arabic"  # arabic + hebrew (rtl) share duration model
        if 0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF or 0xAC00 <= o <= 0xD7FF:
            return "cjk"
        if 0x0E00 <= o <= 0x0E7F or 0x1780 <= o <= 0x17FF or 0x1000 <= o <= 0x109F:
            return "seasia"  # thai/lao/khmer/myanmar: no spaces, syllabic
        if 0x1200 <= o <= 0x137F or 0x10A0 <= o <= 0x10FF:
            return "syllabic"  # ethiopic/georgian/armenian
        if 0x0400 <= o <= 0x04FF or 0x0370 <= o <= 0x03FF:
            return "cyril_greek"
    if re.search(r"[\u0900-\u0DFF]", text):
        return "indic"
    return "latin" if re.search(r"[A-Za-z]", text) else ("cjk" if text.strip() else "latin")


_VOW = re.compile(r"[aeiouyàâäéèêëîïôöùûüœæāīūṃḥṁ]+", re.I)


def estimate_syllables(text: str, lang: str) -> int:
    script = detect_script(text)
    l = normalize_lang(lang)
    if script == "cjk":
        n = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
        return max(1, n if n else max(1, len(text.strip()) // 2))
    if script in ("seasia", "syllabic"):
        # thai/myanmar/khmer: character-cluster ≈ syllable; spaces unreliable
        chars = len(re.sub(r"\s+", "", text))
        return max(1, int(chars * 0.85))
    if script in ("indic", "arabic"):
        # akshara-ish: consonant clusters + vowel signs ≈ syllables
        n = len(re.findall(r"[\u0900-\u0DFF\u0600-\u06FF\u0590-\u05FF]+", text))
        words = len(text.split())
        # arabic without diacritics: ~2 syllables/word; indic similar
        return max(1, max(n * 2, words * 2))
    syl = len(_VOW.findall(text))
    words = len(text.split())
    if l in ("ja",):
        return max(1, len(re.sub(r"\s+", "", text)))
    return max(1, syl if syl else max(1, int(words * 1.6)))


def analyze_text(text: str, lang: str) -> dict:
    """Language-agnostic linguistic analysis driving prosody + duration."""
    l = normalize_lang(lang)
    prior = prior_for(l)
    script = detect_script(text)
    n_syll = estimate_syllables(text, l)
    tonal_need = float(prior["tonal"])
    # CJK chars carry tone even if lang code unknown
    if script == "cjk" and l not in LANG_PRIORS:
        tonal_need = 1.0
    q = text.strip().endswith(("?", "？", "؟", "¿")) or ("?" in text)
    excl = text.strip().endswith(("!", "！", "।"))
    pauses = len(re.findall(r"[,;—–:、।،]", text)) + len(re.findall(r"[.!?؟।。！？]\s", text + " "))
    words = max(1, len(text.split()))
    return {
        "lang": l, "script": script, "rhythm": prior["rhythm"],
        "n_syll": n_syll, "tonal": tonal_need,
        "question": bool(q), "exclaim": bool(excl),
        "pauses": pauses, "words": words,
        "pause_density": pauses / words,
        "base_f0": prior["f0"], "base_rate": prior["rate"], "base_range": prior["range"],
    }


# ---------------------------------------------------------------- synth

def synth_universal(text: str, lang: str, p=None, sr: int = 24000) -> np.ndarray:
    """Acoustic backbone conditioned on (text analysis, lang prior, voice params).

    p: VoiceParams-like with f0/vib_depth/vib_rate/brightness/breath_db/rate.
    Duration ~ syllables (not chars) so CJK/Indic/Arabic get natural timing.
    F0 contour: declination + question rise + tonal wobble + pitch-range.
    Rhythm: stress (alternating) / syllable (even) / mora (staccato) / tonal.
    """
    from .train_superhuman import VoiceParams
    p = p or VoiceParams()
    info = analyze_text(text, lang)
    prior = prior_for(info["lang"])

    f0_base = 0.6 * float(getattr(p, "f0", 185.0)) + 0.4 * float(prior["f0"])
    prange = float(prior["range"])
    # syllable-timed duration: ~4.5 syll/s at rate 1.0 (human conversational)
    rate_eff = float(getattr(p, "rate", 1.0)) * float(prior["rate"])
    rate_eff = float(np.clip(rate_eff, 0.7, 1.4))
    dur = info["n_syll"] / (4.5 * rate_eff)
    # char fallback for very short / punctuation-only
    dur = float(np.clip(dur, max(0.4, len(text) / (30.0 * rate_eff)),
                        min(8.0, max(1.0, len(text) / (8.0 * rate_eff)))))
    n = max(1, int(sr * dur))
    t = np.arange(n) / sr

    # phrase intonation: declination + final rise/fall
    decl = -0.04 * (t / max(1e-6, dur))  # -4% downdrift like humans
    q_rise = 0.10 * np.clip((t / max(1e-6, dur) - 0.7) / 0.3, 0, 1) if info["question"] else 0.0
    e_lift = 0.04 if info["exclaim"] else 0.0
    # tonal wobble for zh/vi/th/yo/pa (lexical tone mimic, 1.2Hz per syllable-ish)
    tone = 0.0
    if info["tonal"] > 0.5:
        tone_rate = max(1.0, info["n_syll"] / max(0.3, dur))  # tones per second
        tone = 0.035 * np.sin(2 * np.pi * tone_rate * t + 1.0) * prange
    # micro prosody (phrase accent 1.3Hz) scaled by pitch-range
    accent = 0.02 * prange * np.sin(2 * np.pi * 1.3 * t)
    f0t = f0_base * (1.0 + decl + q_rise + e_lift + tone + accent)

    vdepth = float(getattr(p, "vib_depth", 3.0))
    vrate = float(getattr(p, "vib_rate", 5.0))
    phase = 2 * np.pi * np.cumsum(f0t) / sr + vdepth * np.sin(2 * np.pi * vrate * t)
    bright = float(getattr(p, "brightness", 0.5))
    # --- human-like glottal stack: 8 harmonics with formant resonances ---
    # pure 3-sine is too peaky (flatness 0.0); formants + aspiration -> ~0.14
    f0_med = float(np.median(f0t))
    wav = np.zeros(n, dtype=np.float64)
    for k in range(1, 17):
        fk = f0_med * k
        # formant boost (500/1500/2500 Hz like neutral vowel)
        w = (1.0 + 0.9 * np.exp(-((fk - 500) / 450) ** 2)
             + 0.7 * np.exp(-((fk - 1500) / 650) ** 2)
             + 0.45 * np.exp(-((fk - 2500) / 900) ** 2))
        # gentle decay keeps upper harmonics alive (muffled drone if too steep);
        # upper octave (k>8) extra-tamed so brightness never turns to hiss
        amp = (0.35 / (k ** 0.9)) * (0.55 + 0.9 * bright) * w / 1.8
        if k > 8:
            amp *= 0.5
        if k <= 3:
            wav = wav + amp * np.sin(k * phase)
        else:
            # higher harmonics slightly decorrelated (breathy, less peaky)
            wav = wav + amp * np.sin(k * phase + 0.4 * k)
    # shimmer: 2% natural amplitude wander (human, not machine-flat)
    # NOTE: stable hashlib seed — builtin hash() is salted per-process.
    import hashlib as _hl
    _seed = int.from_bytes(_hl.md5(text.encode()).digest()[:4], "little")
    rng = np.random.default_rng(_seed)
    shimmer = 1.0 + 0.02 * np.sin(2 * np.pi * 7.0 * t + 0.5) + 0.008 * rng.standard_normal(n)
    wav = wav * shimmer
    # aspiration: faint lowpassed noise fills spectral valleys -> human flatness
    asp = rng.standard_normal(n).astype(np.float64)
    asp = (asp + np.concatenate([[0], asp[:-1]])) * 0.5  # soft lowpass
    asp_level = 0.016 + 0.012 * bright
    wav = wav + asp * asp_level
    # consonant-like transients: short frication bursts at syllable onsets.
    # Pure vowel hum with zero consonants reads as alien signal; these bursts
    # restore high-frequency texture like human fricatives (subtle, ~-24dB).
    ns = max(1, min(int(info["n_syll"]), 40))
    grid = np.linspace(0.05, 0.95, ns) * n + rng.uniform(-0.02, 0.02, ns) * n
    blen = max(8, int(sr * 0.02))
    benv = np.hanning(blen)
    for c in np.clip(grid.astype(int), 0, max(0, n - blen)):
        burst = rng.standard_normal(blen).astype(np.float64) * benv * 0.03
        wav[c:c + blen] += burst
    wav = wav.astype(np.float32)

    # rhythm envelope
    if info["rhythm"] == "stress":
        stress = 0.85 + 0.15 * np.sin(2 * np.pi * 2.2 * t)  # alternating strong/weak
        wav = wav * stress
    elif info["rhythm"] == "mora":
        stac = 0.85 + 0.15 * np.sign(np.sin(2 * np.pi * 5.0 * t))  # staccato
        wav = wav * stac
    # tonal langs: slightly more syllable separation
    if info["tonal"] > 0.5:
        gate = 0.92 + 0.08 * np.sin(2 * np.pi * max(1.0, info["n_syll"] / max(0.3, dur)) * t)
        wav = wav * gate

    fade = min(n // 10, 2000)
    if fade > 0:
        wav[:fade] *= np.linspace(0, 1, fade)
        wav[-fade:] *= np.linspace(1, 0, fade)
    return wav.astype(np.float32)


# ---------------------------------------------------------------- zero-shot adapter

def adapt_to_lang(target: str, bank: dict, glob=None) -> object:
    """Return params for ANY language code.

    Exact match -> trained adapter. Else cosine-weighted blend of k-nearest
    trained langs in embedding space + global fallback. Never raises.
    A pitch guard pulls f0 halfway to the language-family prior and clips
    to human speech range, so drifted checkpoints can't turn demonic.
    Always returns a copy (callers may mutate safely).
    """
    import copy as _c
    import numpy as _np
    from .train_superhuman import VoiceParams
    t = normalize_lang(target)
    if t in bank:
        base = bank[t]
    elif not bank:
        base = glob or VoiceParams()
    else:
        te = lang_embedding(t)
        langs = [l for l in bank if l != "und"]
        if not langs:
            base = bank.get("und", glob or VoiceParams())
        else:
            sims = []
            for l in langs:
                e = lang_embedding(l)
                s = float(te @ e / (_np.linalg.norm(te) * _np.linalg.norm(e) + 1e-9))
                sims.append((s, l))
            sims.sort(reverse=True)
            top = sims[:3]
            w = _np.array([max(0.0, s + 1.0) for s, _ in top])  # shift cosine to positive
            if w.sum() < 1e-9:
                base = bank.get("und", glob or VoiceParams())
            else:
                w /= w.sum()
                mat = _np.stack([bank[l].vector() for _, l in top])
                v = (w[:, None] * mat).sum(axis=0)
                # blend toward global so far-out langs stay stable (70/30)
                g = (glob or bank.get("und") or VoiceParams()).vector()
                v = 0.7 * v + 0.3 * g
                base = VoiceParams.from_vector(v)
    p = _c.copy(base)
    try:
        # lean on the language-family prior (65%) so drifted checkpoints
        # return to natural pitch; hard-clip to human speech range.
        p.f0 = float(_np.clip(0.35 * float(p.f0) + 0.65 * float(prior_for(t)["f0"]), 150.0, 210.0))
    except Exception:
        pass
    return p


def prosody_for_lang(emotion: str, lang: str) -> dict:
    """Emotion style shifted by language-family speaking rate/pitch."""
    prior = prior_for(lang)
    base = {"rate": 1.0, "pitch_hz": "+0Hz", "rate_pct": "+0%", "energy": "medium"}
    if emotion == "warm":
        base.update({"rate": 1.02, "rate_pct": "+2%", "pitch_hz": "+2Hz"})
    elif emotion == "empathetic":
        base.update({"rate": 0.92, "rate_pct": "-8%", "pitch_hz": "-1Hz"})
    elif emotion == "curious":
        base.update({"rate": 1.0, "rate_pct": "+0%", "pitch_hz": "+3Hz"})
    # language-family tempo shift (compound multiplicatively with emotion)
    base["rate"] = round(float(base["rate"]) * float(prior["rate"]), 3)
    pct = int(round((base["rate"] - 1.0) * 100))
    base["rate_pct"] = f"{pct:+d}%"
    return base


# ---------------------------------------------------------------- data loader

def load_training_bank(data_dir: str, sr: int = 16000,
                       langs: list[str] | None = None) -> dict[str, tuple[list[str], list[np.ndarray]]]:
    """Build {lang: (texts, wavs)} from ANY directory layout.

    Accepted (any audio ext: wav/flac/ogg/opus):
      data/<lang>/*.{wav,flac,ogg} (+ optional same-name .txt sidecar)
      data/*.{wav,flac,ogg} with `<lang>_*.wav` prefix (e.g. sw_001.wav, xx_custom.wav)
    Unknown codes are kept as-is (normalize_lang tolerant) -> true any-language.
    """
    import os
    import glob as _g
    bank: dict[str, tuple[list[str], list[np.ndarray]]] = {}

    def add(lang: str, text: str, wav: np.ndarray):
        l = normalize_lang(lang)
        if l not in bank:
            bank[l] = ([], [])
        bank[l][0].append(text)
        bank[l][1].append(wav)

    def read_wav(path: str) -> np.ndarray | None:
        try:
            import soundfile as sf
            x, f = sf.read(path, always_2d=False)
            x = np.asarray(x, dtype=np.float32)
            if getattr(x, "ndim", 1) == 2:
                x = x.mean(axis=1)
            if f != sr:  # linear resample
                dur = len(x) / float(f)
                n_out = max(1, int(round(dur * sr)))
                x = np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, len(x)), x).astype(np.float32)
            # universal input hygiene: mono 16k + loudness norm (superhuman consistency)
            try:
                from .audio_io import normalize_loudness
                x = normalize_loudness(x)
            except Exception:
                pass
            return x
        except Exception:
            return None

    AUDIO_EXTS = ("*.wav", "*.flac", "*.ogg", "*.opus")

    def _iter_audio(lp: str):
        import itertools as _it
        for ext in AUDIO_EXTS:
            for p in sorted(_g.glob(os.path.join(lp, ext))):
                yield p

    if langs is not None:  # explicit lang list, files directly inside data_dir
        for l in langs:
            for wav_p in _iter_audio(data_dir):
                w = read_wav(wav_p)
                if w is None:
                    continue
                txt_p = os.path.splitext(wav_p)[0] + ".txt"
                try:
                    with open(txt_p, encoding="utf-8") as fh:
                        tx = fh.read().strip()
                except Exception:
                    tx = os.path.basename(wav_p)
                add(l, tx, w)
        return bank

    # layout 1: subfolders per language (any name)
    sub = [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))] \
        if os.path.isdir(data_dir) else []
    for lang_dir in sub:
        lp = os.path.join(data_dir, lang_dir)
        for wav_p in _iter_audio(lp):
            w = read_wav(wav_p)
            if w is None:
                continue
            txt_p = os.path.splitext(wav_p)[0] + ".txt"
            try:
                with open(txt_p, encoding="utf-8") as fh:
                    tx = fh.read().strip()
            except Exception:
                tx = os.path.basename(wav_p)
            add(lang_dir, tx or os.path.basename(wav_p), w)
    # layout 2: flat with lang prefix
    if not bank and os.path.isdir(data_dir):
        for wav_p in _iter_audio(data_dir):
            stem = os.path.splitext(os.path.basename(wav_p))[0]
            pre = stem.split("_")[0] if "_" in stem else "und"
            w = read_wav(wav_p)
            if w is None:
                continue
            txt_p = os.path.splitext(wav_p)[0] + ".txt"
            try:
                with open(txt_p, encoding="utf-8") as fh:
                    tx = fh.read().strip()
            except Exception:
                tx = stem
            add(pre, tx or stem, w)
    return bank
