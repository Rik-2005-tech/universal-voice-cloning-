"""STT / LLM / TTS backends. Dummy backends run offline; real ones lazy-import."""
from __future__ import annotations
import asyncio
import re
from dataclasses import dataclass


# ---------- abstract ----------

class STTBackend:
    lang: str = "en"
    def transcribe(self, wav16k, sr: int = 16000) -> tuple[str, str, float]:
        """Return (text, lang_code, confidence)."""
        raise NotImplementedError

class LLMBackend:
    def complete(self, text: str, lang: str) -> str:
        raise NotImplementedError
    async def stream(self, text: str, lang: str):
        yield self.complete(text, lang)

class TTSBackend:
    sr: int = 24000
    def synth(self, text: str, lang: str, style: dict | None = None) -> tuple[object, int]:
        raise NotImplementedError


# ---------- dummy (offline, deterministic) ----------

@dataclass
class DummySTT(STTBackend):
    """Keyword-based LID demo: no model download needed."""
    def transcribe(self, wav16k, sr: int = 16000):
        import numpy as np
        x = __import__("numpy").asarray(wav16k, dtype=float)
        loud = float((x ** 2).mean() ** 0.5) if len(x) else 0.0
        if loud < 1e-4:
            return "", "en", 0.0
        # In real use faster-whisper returns text+lang. Here fixed multilingual demo.
        return "bonjour, comment vas-tu ?", "fr", 0.9


@dataclass
class DummyLLM(LLMBackend):
    REPLIES = {
        "fr": "Bonjour ! Je vais très bien, merci. Et vous, comment puis-je vous aider aujourd'hui ?",
        "en": "Hello! I'm doing great, thanks. How can I help you today?",
        "es": "¡Hola! Estoy muy bien, gracias. ¿Cómo puedo ayudarte hoy?",
        "de": "Hallo! Mir geht es sehr gut, danke. Wie kann ich Ihnen heute helfen?",
        "hi": "नमस्ते! मैं बहुत अच्छा हूँ, धन्यवाद। आज मैं आपकी कैसे मदद कर सकता हूँ?",
    }
    def complete(self, text: str, lang: str) -> str:
        base = self.REPLIES.get(lang, self.REPLIES["en"])
        # echo intent: keep it clean, grammatical, disfluency-free => better than human
        short = re.sub(r"\s+", " ", text).strip()[:120]
        if short:
            return f"{base} (Vous avez dit : « {short} ».)"
        return base
    async def stream(self, text: str, lang: str):
        full = self.complete(text, lang)
        # stream word-by-word to simulate LLM tokens for low-latency TTS
        for w in full.split(" "):
            yield w + " "
            await asyncio.sleep(0)


@dataclass
class DummyTTS(TTSBackend):
    sr = 24000
    def synth(self, text: str, lang: str, style: dict | None = None):
        """Sine-based placeholder audio whose duration ~ text length. Replace with real TTS."""
        import numpy as np
        style = style or {}
        rate = float(style.get("rate", 1.0))
        dur = max(0.4, min(8.0, len(text) / (14.0 * rate)))
        n = int(self.sr * dur)
        t = np.arange(n) / self.sr
        f0 = {"fr": 196.0, "en": 180.0, "es": 200.0}.get(lang, 185.0)
        # gentle vibrato + fade = pleasant superhuman steadiness
        vib = 3.0 * np.sin(2 * np.pi * 5.0 * t)
        wav = 0.35 * np.sin(2 * np.pi * (f0 * t) + vib)
        fade = min(n // 10, 2000)
        wav[:fade] *= np.linspace(0, 1, fade)
        wav[-fade:] *= np.linspace(1, 0, fade)
        return wav.astype(np.float32), self.sr


# ---------- trained superhuman voice (indistinguishable head) ----------

@dataclass
class SuperhumanTTS(TTSBackend):
    """Universal any-language voice: loads voice.npz (single) or bank.npz
    (langs+mat from train_multilingual). Unseen langs -> zero-shot blend
    via universal.adapt_to_lang. Synth uses language-aware prosody so every
    language gets native rhythm/intonation + humanize texture."""
    ckpt: str | None = None
    sr: int = 24000
    def _bank(self):
        import numpy as np
        from .train_superhuman import VoiceParams, load_bank
        if self.ckpt:
            try:
                b = load_bank(self.ckpt)
                g = b.get("und", VoiceParams())
                return b, g
            except Exception:
                pass
        from .train_superhuman import VoiceParams as _V
        g = _V()
        return {"und": g}, g
    def _params(self):
        from .train_superhuman import VoiceParams
        b, g = self._bank()
        # default lang handled per-synth; return global for compat
        return b.get("und", g)
    def synth(self, text: str, lang: str, style: dict | None = None):
        from .universal import synth_universal, adapt_to_lang
        from .humanize import humanize
        bank, glob = self._bank()
        p = adapt_to_lang(lang, bank, glob)
        if style and "rate" in style:
            import copy as _c
            p = _c.copy(p)
            p.rate = float(style["rate"])
        wav = synth_universal(text, lang, p, self.sr)
        # language-family base f0 for jitter reference
        from .universal import prior_for
        f0ref = prior_for(lang)["f0"]
        return humanize(wav, self.sr, text, f0=f0ref), self.sr


# ---------- voice map for real TTS (edge-tts, 100+ langs) ----------

EDGE_VOICE = {
    "en": "en-US-AriaNeural", "fr": "fr-FR-DeniseNeural", "es": "es-ES-ElviraNeural",
    "de": "de-DE-KatjaNeural", "it": "it-IT-ElsaNeural", "pt": "pt-BR-FranciscaNeural",
    "hi": "hi-IN-SwaraNeural", "ar": "ar-SA-ZariyahNeural", "zh": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural", "ko": "ko-KR-SunHiNeural", "ru": "ru-RU-SvetlanaNeural",
    "nl": "nl-NL-ColetteNeural", "pl": "pl-PL-ZofiaNeural", "tr": "tr-TR-EmelNeural",
    "ta": "ta-IN-PallaviNeural", "te": "te-IN-ShrutiNeural", "mr": "mr-IN-AarohiNeural",
    "bn": "bn-IN-TanishaaNeural", "ml": "ml-IN-SobhanaNeural", "kn": "kn-IN-SapnaNeural",
    "gu": "gu-IN-DhwaniNeural", "pa": "pa-IN-NeerjaNeural", "ur": "ur-PK-AsmaNeural",
    "id": "id-ID-GadisNeural", "ms": "ms-MY-YasminNeural", "vi": "vi-VN-HoaiMyNeural",
    "th": "th-TH-PremwadeeNeural", "uk": "uk-UA-PolinaNeural", "el": "el-GR-AthinaNeural",
}
def voice_for(lang: str) -> str:
    return EDGE_VOICE.get(lang, EDGE_VOICE["en"])


# ---------- real backends (optional, lazy imports) ----------

class FasterWhisperSTT(STTBackend):
    """Streaming-capable multilingual ASR with built-in LID. Requires faster-whisper."""
    def __init__(self, model: str = "small", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(model, device=device, compute_type=compute_type)
    def transcribe(self, wav16k, sr: int = 16000):
        import numpy as np
        x = np.asarray(wav16k, dtype=np.float32)
        segs, info = self.m.transcribe(x, beam_size=1, vad_filter=True)
        txt = " ".join(s.text.strip() for s in segs).strip()
        return txt, (info.language or "en"), float(info.language_probability or 0.0)


class EdgeTTSBackend(TTSBackend):
    sr = 24000
    def synth(self, text: str, lang: str, style: dict | None = None):
        import asyncio as _a, io
        import edge_tts
        style = style or {}
        rate = style.get("rate_pct", "+0%")
        pitch = style.get("pitch_hz", "+0Hz")
        ssml_rate = rate if isinstance(rate, str) else f"{int((float(rate)-1)*100):+d}%"
        async def _run():
            c = edge_tts.Communicate(text, voice_for(lang), rate=ssml_rate, pitch=pitch)
            buf = io.BytesIO()
            async for chunk in c.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()
        mp3 = _a.run(_run())
        # decode mp3 -> wav needs ffmpeg/soundfile; return raw bytes marker if unavailable
        return mp3, 0  # sr=0 signals encoded bytes; server handles it


class TransformersLLM(LLMBackend):
    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.net = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="auto")
        import torch as _t
        self._t = _t
    def complete(self, text: str, lang: str) -> str:
        sys = f"Reply ONLY in ISO language '{lang}'. Be concise, warm, perfectly grammatical, one breath."
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": text}]
        ids = self.tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(self.net.device)
        out = self.net.generate(ids, max_new_tokens=120, do_sample=False)
        return self.tok.decode(out[0][len(ids[0]):], skip_special_tokens=True)
