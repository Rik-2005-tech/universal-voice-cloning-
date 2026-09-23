"""Human-vs-AI voice detector: acoustic features + logistic regression.

Trained on REAL data: human class = LibriSpeech/GramVaani/SLR speech + your
recordings; AI class = Piper neural voices + PolyVoice synth in 7 languages.
Numpy-only (no sklearn needed). See train_spoof.py.
"""
from __future__ import annotations
import numpy as np

FEATURES = ["flat_med", "flat_std", "hf2k", "hf4k", "centroid", "rolloff",
            "zcr", "voiced_ratio", "jitter", "rms_dyn_db", "flux", "pauses",
            "crest", "flat_high", "hf6k", "contrast",
            "gd_var", "hf_slope", "hf_burst"]


FALLBACK_TEXTS = {
    "en": ["Hello! How are you today?", "The weather is wonderful this morning.",
           "Good morning, welcome aboard.", "Thank you very much, goodbye!"],
    "hi": ["नमस्ते! आप कैसे हैं?", "आज मौसम बहुत अच्छा है।",
           "बहुत-बहुत धन्यवाद, अलविदा!", "आपका नाम क्या है?"],
    "bn": ["নমস্কার! আপনি কেমন আছেন?", "আজ আবহাওয়া খুব ভালো।",
           "অনেক অনেক ধন্যবাদ, বিদায়!", "আপনার নাম কী?"],
    "fr": ["Bonjour, comment vas-tu?", "Je vais très bien, merci beaucoup.",
           "Merci beaucoup, au revoir!", "Quel temps fait-il aujourd'hui?"],
    "es": ["Hola, cómo estás hoy?", "Muchas gracias, adiós!",
           "El clima está maravilloso hoy.", "Hasta luego, que tengas buen día."],
    "zh": ["你好，你今天怎么样?", "今天天气非常好。", "非常感谢，再见！", "请问你叫什么名字?"],
    "ar": ["مرحبا، كيف حالك اليوم؟", "شكرا جزيلا، وداعا!",
           "الطقس جميل جدا هذا الصباح.", "إلى اللقاء، أتمنى لك يوما سعيدا."],
}
EXTRA_TEXTS = {
    "es": ["Hola, cómo estás hoy?", "Muchas gracias, adiós!",
           "El clima está maravilloso hoy.", "Hasta luego, que tengas buen día.",
           "Buenos días, bienvenido a casa.", "Dónde está la estación de tren?",
           "Necesito ayuda urgente, por favor.", "Me gusta mucho esta canción.",
           "Qué hora es ahora mismo?", "Hablas muy claro, gracias.",
           "Ayer llovió todo el día.", "Mañana será un día soleado.",
           "Mi familia vive en un pueblo pequeño.", "El libro es muy interesante.",
           "Por favor llama un taxi.", "La cena está lista, vamos.",
           "Tengo una pregunta importante.", "Nos vemos pronto, cuídate."],
    "ar": ["مرحبا، كيف حالك اليوم؟", "شكرا جزيلا، وداعا!",
           "الطقس جميل جدا هذا الصباح.", "إلى اللقاء، أتمنى لك يوما سعيدا.",
           "صباح الخير، أهلا بك.", "أين محطة القطار من فضلك؟",
           "أحتاج إلى مساعدة عاجلة.", "أحب هذه الأغنية كثيرا.",
           "كم الساعة الآن؟", "كلامك واضح جدا، شكرا.",
           "أمطرت السماء طوال أمس.", "غدا سيكون يوما مشمسا.",
           "عائلتي تعيش في قرية صغيرة.", "الكتاب ممتع جدا.",
           "من فضلك اتصل بسيارة أجرة.", "العشاء جاهز، هيا بنا.",
           "لدي سؤال مهم.", "أراك قريبا، اعتن بنفسك."],
}


def extract_features(wav, sr: int = 16000) -> np.ndarray:
    """12-dim voice fingerprint. Never raises (zeros on failure)."""
    try:
        from .audio_io import to_mono_16k
        from .timbre import f0_stats
        x = to_mono_16k(np.asarray(wav), int(sr or 16000), 16000).astype(np.float64)
        sr = 16000
        if len(x) < 2048:
            x = np.pad(x, (0, 2048 - len(x)))
        n_fft, hop = 1024, 256
        frames = [x[i:i + n_fft] * np.hanning(n_fft)
                  for i in range(0, len(x) - n_fft + 1, hop)]
        S = np.stack(frames)
        mag = np.abs(np.fft.rfft(S, axis=1)) + 1e-9
        F = mag.shape[1]
        freqs = np.fft.rfftfreq(n_fft, 1 / sr)
        rms = np.sqrt((mag ** 2).mean(axis=1) + 1e-12)
        voiced = rms > (np.median(rms) * 0.5)
        if voiced.sum() < 4:
            voiced = np.ones(len(rms), dtype=bool)
        M = mag[voiced]
        # flatness per voiced frame
        flats = np.exp(np.log(M).mean(axis=1)) / M.mean(axis=1)
        # spectral shape
        tot = M.sum(axis=1, keepdims=True)
        tot1 = tot.ravel()
        cent = (M * freqs).sum(axis=1) / tot1
        cum = np.cumsum(M / tot, axis=1)
        roll = freqs[np.argmax(cum > 0.85, axis=1)]
        hf2k = (M[:, freqs > 2000].sum(axis=1) / tot1)
        hf4k = (M[:, freqs > 4000].sum(axis=1) / tot1)
        # zero-crossing rate
        z = np.sign(x)
        zcr = float(np.mean(np.abs(np.diff(z)) > 0) / 2)
        # dynamics + flux
        lrms = 20 * np.log10(rms + 1e-9)
        dyn = float(np.percentile(lrms, 90) - np.percentile(lrms, 10))
        flux = float(np.mean(np.abs(np.diff(np.log(M + 1e-9), axis=0))))
        pauses = float(1.0 - voiced.mean())
        # vocoder-artifact band: crest + flatness + energy above 6kHz,
        # plus spectral contrast (neural audio often over-smooths valleys)
        lM = 20 * np.log10(M + 1e-9)
        crest = float(np.median(M.max(axis=1) / (M.mean(axis=1) + 1e-9)))
        hi = M[:, freqs > 4000]
        lhi = 20 * np.log10(hi + 1e-9)
        flat_high = float(np.median(np.exp(lhi.mean(axis=1)) / (hi.mean(axis=1) + 1e-9)))
        hf6k = float(np.median(M[:, freqs > 6000].sum(axis=1) / tot1))
        contrast = float(np.median(np.percentile(lM, 90, axis=1)
                                   - np.percentile(lM, 10, axis=1)))
        # short-window vocoder traces (work even in <4s clips):
        # gd_var = group-delay roughness (AI phase stumbles frame to frame)
        ph = np.angle(np.fft.rfft(S[voiced], axis=1))
        gd = np.diff(np.unwrap(ph, axis=1), axis=1)
        band = (freqs[1:] > 300) & (freqs[1:] < 6000)
        gd_var = float(np.median(np.std(gd[:, band], axis=1)) + 1e-9)
        gd_var = float(np.log1p(gd_var))
        # hf_slope = high-band tilt (vocoder hiss tilts differently)
        hb = (freqs > 4000) & (freqs < 8000)
        lhb = 20 * np.log10(M[:, hb] + 1e-9)
        xs = np.linspace(-1, 1, lhb.shape[1])
        hf_slope = float(np.median([(l - l.mean()) @ xs / (xs @ xs) for l in lhb]))
        # hf_burst = burstiness of high-band energy (humans burst, bots hiss flat)
        hfe = np.log10(M[:, freqs > 6000].sum(axis=1) + 1e-9)
        hf_burst = float(np.std(hfe))
        fs = f0_stats(x, sr)
        vr = float(fs["voiced"] / max(1, len(frames)))
        feat = np.array([float(np.median(flats)), float(np.std(flats)),
                         float(np.median(hf2k)), float(np.median(hf4k)),
                         float(np.median(cent)), float(np.median(roll)),
                         zcr, vr, float(fs["jitter_rel"]), dyn, flux, pauses,
                         crest, flat_high, hf6k, contrast,
                         gd_var, hf_slope, hf_burst])
        return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    except Exception:
        return np.zeros(len(FEATURES))


def train_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                 iters: int = 3000, lr: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = rng.standard_normal(Xs.shape[1]) * 0.01
    b = 0.0
    n = len(y)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        err = p - y
        w -= lr * (Xs.T @ err / n + l2 * w / n)
        b -= lr * err.mean()
    return w, float(b), mu, sd


def predict_proba(X: np.ndarray, w, b, mu, sd) -> np.ndarray:
    Xs = (np.asarray(X, dtype=np.float64) - mu) / sd
    return 1.0 / (1.0 + np.exp(-(Xs @ w + b)))


def save_detector(path: str, w, b, mu, sd, acc: float,
                  thr_short: float = 0.5, thr_long: float = 0.5,
                  thr_map: dict | None = None) -> None:
    keys = sorted((thr_map or {}).keys())
    np.savez(path, w=w, b=np.array(b), mu=mu, sd=sd,
             acc=np.array(acc), feats=np.array(FEATURES),
             thr_short=np.array(thr_short), thr_long=np.array(thr_long),
             thr_keys=np.array(keys),
             thr_vals=np.array([(thr_map or {})[k] for k in keys]))


def load_detector(path: str = "detect.npz"):
    d = np.load(path, allow_pickle=True)
    w = np.asarray(d["w"], dtype=np.float64).ravel()
    mu = np.asarray(d["mu"], dtype=np.float64).ravel()
    sd = np.asarray(d["sd"], dtype=np.float64).ravel()
    # backward compat: old 12-dim weights work with 16+/19-dim features
    # (new artifact features get zero weight until retrained)
    n = len(FEATURES)
    if w.size < n:
        w = np.pad(w, (0, n - w.size))
        mu = np.pad(mu, (0, n - mu.size))
        sd = np.pad(sd, (0, n - sd.size), constant_values=1.0)
    thr_s = float(d["thr_short"]) if "thr_short" in d.files else 0.5
    thr_l = float(d["thr_long"]) if "thr_long" in d.files else 0.5
    tmap = {}
    if "thr_keys" in d.files and len(d["thr_keys"]):
        tmap = {str(k): float(v) for k, v in zip(d["thr_keys"], d["thr_vals"])}
    return w, float(d["b"]), mu, sd, float(d["acc"]), thr_s, thr_l, tmap


def _score_once(x, sr, w, b, mu, sd) -> float:
    f = extract_features(x, sr)
    return float(predict_proba(f[None, :], w, b, mu, sd)[0])


def _model_path_for(lang: str, path: str) -> str:
    """Per-language weights when present (detect_hi.npz etc.), else the
    shared model. Per-lang models resolve the bright/dark contradiction
    between languages' human recordings."""
    try:
        import os as _os
        try:
            from .universal import normalize_lang as _nl
            l = _nl(lang)
        except Exception:
            l = (lang or "en").split("-")[0].lower()
        if l in ("hi", "en", "bn"):
            cand = _os.path.join(_os.path.dirname(path) or ".", f"detect_{l}.npz")
            if _os.path.exists(cand):
                return cand
    except Exception:
        pass
    return path


_LID_MODEL = None


def detect_language_audio(wav, sr: int = 16000) -> tuple[str, float]:
    """Identify the spoken language straight from audio (faster-whisper LID).
    Returns (lang_code, confidence). Never raises (falls back to en, 0.0)."""
    global _LID_MODEL
    try:
        import numpy as _np
        from .audio_io import to_mono_16k
        x = to_mono_16k(_np.asarray(wav), int(sr or 16000), 16000).astype(_np.float32)
        if _LID_MODEL is None:
            from faster_whisper import WhisperModel
            try:
                _LID_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
            except Exception:
                _LID_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
        segs, info = _LID_MODEL.transcribe(x, beam_size=1, vad_filter=True)
        # drain generator so language detecting actually runs
        for _ in segs:
            pass
        return (info.language or "en"), float(info.language_probability or 0.0)
    except Exception:
        return "en", 0.0


def verdict_auto(wav, sr: int = 16000, path: str = "detect.npz") -> dict:
    """Detect the language from the audio, then call that language's model
    (per-lang weights for hi/en/bn, shared model otherwise). Low-confidence
    LID (<0.5) falls back to the shared model instead of routing blindly.
    Returns verdict dict + detected_lang. Never raises."""
    try:
        lang, conf = detect_language_audio(wav, sr)
        if conf < 0.5:
            v = verdict(wav, sr, path)
            v["detected_lang"] = lang
            v["lid_conf"] = round(conf, 2)
            v["routing"] = "global(low-lid-conf)"
            return v
        v = verdict(wav, sr, path, lang=lang)
        v["detected_lang"] = lang
        v["lid_conf"] = round(conf, 2)
        return v
    except Exception as e:
        return {"label": "UNKNOWN", "p_ai": -1.0, "error": str(e)[:100]}


def verdict(wav, sr: int = 16000, path: str = "detect.npz", lang: str | None = None) -> dict:
    """Condition-aware verdict: multi-window median score (robust to local
    noise bursts) + per-condition thresholds (clean/noisy/crowded x
    short/long). Routes hi/en/bn to per-language weights when present.
    Returns {label, p_ai, threshold, condition}. Never raises."""
    try:
        use_path = _model_path_for(lang, path) if lang else path
        w, b, mu, sd, acc, thr_s, thr_l, tmap = load_detector(use_path)
        x = np.asarray(wav)
        sr = int(sr or 16000)
        dur = len(x) / float(sr)
        cond = estimate_condition(x, sr)["bucket"]
        # multi-window for files >= 6s: median resists local bursts
        if dur >= 6.0:
            wl, hop = 4.0 * sr, 2.0 * sr
            ps = [_score_once(x[int(s):int(s + wl)], sr, w, b, mu, sd)
                  for s in np.arange(0, max(1, len(x) - wl + 1), hop)]
            p = float(np.median(ps)) if ps else _score_once(x, sr, w, b, mu, sd)
            nw = len(ps)
        else:
            p = _score_once(x, sr, w, b, mu, sd)
            nw = 1
        key = f"{'short' if dur < 4.0 else 'long'}_{cond}"
        thr = float(tmap.get(key, thr_s if dur < 4.0 else thr_l))
        import os as _os2
        return {"label": "AI" if p >= thr else "HUMAN", "p_ai": round(p, 3),
                "threshold": round(thr, 2), "model_acc": round(acc, 3),
                "condition": cond, "windows": nw, "model": _os2.path.basename(use_path)}
    except Exception as e:
        return {"label": "UNKNOWN", "p_ai": -1.0, "error": str(e)[:100]}


def phone_channel(wav, sr: int = 16000, snr_db: float = 20.0,
                  seed: int = 0) -> np.ndarray:
    """Simulate a phone call: 8kHz, 300-3400Hz band, faint line noise.

    Applied to BOTH classes so the detector learns fakeness, not channels.
    """
    try:
        rng = np.random.default_rng(seed)
        x = np.asarray(wav, dtype=np.float64).ravel()
        n8 = max(160, int(round(len(x) / float(sr or 16000) * 8000)))
        x8 = np.interp(np.linspace(0, 1, n8), np.linspace(0, 1, len(x)), x)
        S = np.fft.rfft(x8)
        freqs = np.fft.rfftfreq(len(x8), 1 / 8000)
        S[(freqs < 300) | (freqs > 3400)] *= 0.05
        x8 = np.fft.irfft(S, n=len(x8)).real
        sig = float(np.sqrt(np.mean(x8 ** 2) + 1e-12))
        x8 = x8 + rng.standard_normal(len(x8)) * sig / (10.0 ** (snr_db / 20.0))
        x16 = np.interp(np.linspace(0, 1, len(x)), np.linspace(0, 1, len(x8)), x8)
        m = float(np.max(np.abs(x16)) + 1e-12)
        return (x16 * min(1.0, 0.89 / m)).astype(np.float32)
    except Exception:
        return np.asarray(wav, dtype=np.float32)


def add_noise(wav, sr: int = 16000, snr_db: float = 15.0, seed: int = 0) -> np.ndarray:
    """Street/babble-like white noise bed. Both classes."""
    try:
        rng = np.random.default_rng(seed)
        x = np.asarray(wav, dtype=np.float64).ravel()
        sig = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        y = x + rng.standard_normal(len(x)) * sig / (10.0 ** (snr_db / 20.0))
        m = float(np.max(np.abs(y)) + 1e-12)
        return (y * min(1.0, 0.89 / m)).astype(np.float32)
    except Exception:
        return np.asarray(wav, dtype=np.float32)


CALL_NOISES = ("street", "babble", "vehicle")


def add_call_noise(wav, sr: int = 16000, kind: str | None = None,
                   snr_db: float = 10.0, seed: int = 0) -> np.ndarray:
    """Realistic call background: street rumble, crowd babble, or vehicle
    cabin hum. Mixed under the voice like a real phone call. Numpy-only."""
    try:
        rng = np.random.default_rng(seed)
        sr = int(sr or 16000)
        x = np.asarray(wav, dtype=np.float64).ravel()
        n = len(x)
        if kind is None:
            kind = CALL_NOISES[int(rng.integers(0, len(CALL_NOISES)))]
        if kind == "street":
            # lowpassed rumble + hiss (traffic through a mic)
            w = rng.standard_normal(n)
            S = np.fft.rfft(w)
            fr = np.fft.rfftfreq(n, 1 / sr)
            S = S / (1.0 + (fr / 400.0) ** 2)  # heavy lowpass
            noise = np.fft.irfft(S, n=n).real
        elif kind == "babble":
            # crowd: noise gated by slow random syllabic envelope
            w = rng.standard_normal(n)
            env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * np.arange(n) / sr
                                     + rng.uniform(0, 6.28))
            env = env * (0.6 + 0.4 * rng.standard_normal(n))
            S = np.fft.rfft(w * np.clip(env, 0, None))
            fr = np.fft.rfftfreq(n, 1 / sr)
            S[(fr < 200) | (fr > 5000)] *= 0.2  # voice-band crowd
            noise = np.fft.irfft(S, n=n).real
        else:  # vehicle cabin: engine hum + harmonics + soft hiss
            t = np.arange(n) / sr
            f0 = float(rng.uniform(70, 130))
            hum = (np.sin(2 * np.pi * f0 * t)
                   + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
                   + 0.25 * np.sin(2 * np.pi * 3 * f0 * t))
            hum = hum / (np.abs(hum).max() + 1e-9)
            noise = hum * 0.7 + rng.standard_normal(n) * 0.3
        sig = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        nz = float(np.sqrt(np.mean(noise ** 2) + 1e-12))
        y = x + noise * (sig / (nz + 1e-12)) / (10.0 ** (snr_db / 20.0))
        m = float(np.max(np.abs(y)) + 1e-12)
        return (y * min(1.0, 0.89 / m)).astype(np.float32)
    except Exception:
        return np.asarray(wav, dtype=np.float32)


def add_reverb(wav, sr: int = 16000, decay: float = 0.35, seed: int = 0) -> np.ndarray:
    """Small-room reverberation (exponential-decay noise IR). Both classes."""
    try:
        rng = np.random.default_rng(seed)
        x = np.asarray(wav, dtype=np.float64).ravel()
        L = min(int(sr * 0.25), max(16, len(x) // 4))
        ir = rng.standard_normal(L) * np.exp(-np.arange(L) / (L * decay + 1e-9))
        ir = ir / (np.abs(ir).sum() + 1e-9)
        y = np.convolve(x, ir, mode="full")[:len(x)]
        y = 0.7 * y + 0.3 * x
        m = float(np.max(np.abs(y)) + 1e-12)
        return (y * min(1.0, 0.89 / m)).astype(np.float32)
    except Exception:
        return np.asarray(wav, dtype=np.float32)


def speed_perturb(wav, sr: int = 16000, factor: float = 1.0) -> np.ndarray:
    """Tempo change without pitch change (linear resample). Both classes."""
    try:
        x = np.asarray(wav, dtype=np.float64).ravel()
        n = max(16, int(round(len(x) / float(factor))))
        y = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)
        return y.astype(np.float32)
    except Exception:
        return np.asarray(wav, dtype=np.float32)


def mix_voices(a, b, sr: int = 16000, voice_snr_db: float = 0.0,
               seed: int = 0) -> np.ndarray:
    """Cocktail: two voices talking over each other (crowded places)."""
    try:
        rng = np.random.default_rng(seed)
        xa = np.asarray(a, dtype=np.float64).ravel()
        xb = np.asarray(b, dtype=np.float64).ravel()
        n = max(len(xa), len(xb))
        if len(xa) < n:
            xa = np.tile(xa, int(np.ceil(n / len(xa))))[:n]
        if len(xb) < n:
            xb = np.tile(xb, int(np.ceil(n / len(xb))))[:n]
        off = int(rng.integers(0, max(1, n // 4)))
        xb = np.concatenate([np.zeros(off), xb])[:n]
        sa = float(np.sqrt(np.mean(xa ** 2) + 1e-12))
        sb = float(np.sqrt(np.mean(xb ** 2) + 1e-12))
        xb = xb * (sa / (sb + 1e-12)) / (10.0 ** (voice_snr_db / 20.0))
        y = xa + xb
        m = float(np.max(np.abs(y)) + 1e-12)
        return (y * min(1.0, 0.89 / m)).astype(np.float32)
    except Exception:
        return np.asarray(a, dtype=np.float32)


def estimate_condition(wav, sr: int = 16000) -> dict:
    """clean / noisy / crowded + SNR estimate. Never raises."""
    try:
        x = np.asarray(wav, dtype=np.float64).ravel()
        sr = int(sr or 16000)
        fl = max(64, int(sr * 0.05))
        env = np.array([np.sqrt(np.mean(x[i:i + fl] ** 2) + 1e-12)
                        for i in range(0, max(1, len(x) - fl), fl)])
        lo = float(np.percentile(env, 10))
        hi = float(np.percentile(env, 90))
        snr = 20.0 * np.log10(hi / (lo + 1e-12))
        # spectral occupancy: crowded voices fill more bins steadily
        n_fft = 1024
        frames = [x[i:i + n_fft] * np.hanning(min(n_fft, max(1, len(x) - i)))
                  for i in range(0, max(1, len(x) - n_fft), n_fft // 2)]
        S = np.stack(frames) if frames else np.zeros((1, n_fft))
        mag = np.abs(np.fft.rfft(S, axis=1)) + 1e-12
        lmag = 20 * np.log10(mag)
        occ = float(np.mean((lmag > (lmag.mean(axis=1, keepdims=True) + 10)).mean()))
        if occ > 0.28 and snr < 18.0:
            bucket = "crowded"
        elif snr < 12.0:
            bucket = "noisy"
        else:
            bucket = "clean"
        return {"snr_db": round(snr, 1), "occupancy": round(occ, 3), "bucket": bucket}
    except Exception:
        return {"snr_db": 99.0, "occupancy": 0.0, "bucket": "clean"}


def augment(wav, sr: int = 16000, seed: int = 0) -> np.ndarray:
    """Rotate phone/noise/reverb/speed by seed. Matched for both classes."""
    try:
        rng = np.random.default_rng(seed)
        kind = int(rng.integers(0, 4))
        if kind == 0:
            return phone_channel(wav, sr, seed=seed)
        if kind == 1:
            return add_noise(wav, sr, snr_db=float(rng.uniform(10, 20)), seed=seed)
        if kind == 2:
            return add_reverb(wav, sr, decay=float(rng.uniform(0.2, 0.5)), seed=seed)
        return speed_perturb(wav, sr, factor=float(rng.uniform(0.92, 1.08)))
    except Exception:
        return np.asarray(wav, dtype=np.float32)
