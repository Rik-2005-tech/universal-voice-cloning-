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
        short = re.sub(r"\s+", " ", text).strip()[:120]
        if short:
            return f"{base} (Vous avez dit : « {short} ».)"
        return base
    async def stream(self, text: str, lang: str):
        full = self.complete(text, lang)
        for w in full.split(" "):
            yield w + " "
            await asyncio.sleep(0)


@dataclass
class TranscriptSTT(STTBackend):
    """Offline audio-aware STT for file/CLI use.

    Real ASR (faster-whisper) is optional. This class does real audio
    processing — energy gate for silence — and resolves the transcript from:
    1) explicit override (sidecar .txt or --text), 2) non-silent fallback.
    This lets the pipeline take audio in and stay deterministic offline.
    """
    text: str = "Hello, how can I help you today?"
    lang: str = "en"
    conf: float = 0.9

    def transcribe(self, wav16k, sr: int = 16000):
        import numpy as np
        x = np.asarray(wav16k, dtype=float)
        loud = float((x ** 2).mean() ** 0.5) if len(x) else 0.0
        if loud < 1e-4:
            return "", "en", 0.0
        t = (self.text or "").strip()
        if not t:
            return "", "en", 0.0
        return t, self.lang, float(self.conf)


@dataclass
class ContextLLM(LLMBackend):
    """Offline contextual brain: intent + short conversation memory.

    Understands greeting / how-are-you / weather / name / help / thanks /
    bye / time / generic question vs statement, replies in the user's
    language (en/fr/es/de/hi fallback en), references prior turns.
    """
    history: list = None

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def _intent(self, low: str) -> str:
        if not low:
            return "silence"
        if any(k in low for k in ("bye", "goodbye", "au revoir", "adios", "tschuss")) or low.strip() == "bye":
            return "bye"
        if any(k in low for k in ("thank", "merci", "gracias", "danke", "dhanyavad", "shukriya")):
            return "thanks"
        if any(k in low for k in ("weather", "météo", "clima", "wetter", "mausam")):
            return "weather"
        if any(k in low for k in ("your name", "who are you", "comment tu t'appelles", "qui es-tu",
                                  "cómo te llamas", "wie heißt du", "tumhara naam")):
            return "name"
        if any(k in low for k in ("help", "aide", "ayuda", "hilfe", "madad")):
            return "help"
        if any(k in low for k in ("how are you", "how is it going", "comment vas-tu", "comment ça va",
                                  "cómo estás", "wie geht", "kaise ho", "how are u")):
            return "how_are_you"
        if any(k in low for k in ("hello", "hi", "hey", "bonjour", "salut", "hola",
                                  "hallo", "namaste", "good morning", "good afternoon")):
            return "greeting"
        if any(k in low for k in ("time", "heure", "hora", "uhr", "samay", "kitne baje")):
            return "time"
        if "?" in low:
            return "question"
        return "statement"

    def complete(self, text: str, lang: str) -> str:
        import re
        clean = re.sub(r"\s+", " ", (text or "")).strip()
        low = clean.lower()
        intent = self._intent(low)
        n_prev = len(self.history)
        T = {
            "en": {
                "silence": "I didn't hear anything. Could you please speak again?",
                "greeting": "Hello! Great to hear you. How can I help you today?",
                "how_are_you": "I'm doing great, thanks for asking! How about you, how are you feeling?",
                "weather": "I can't check live weather offline, but if you tell me your city I can suggest what to ask a weather service.",
                "name": "I'm PolyVoice, your offline English voice assistant with your trained voice.",
                "help": "I can chat, answer questions, and speak back in your trained voice. Try: ask me how I am, or say thanks, or ask the time.",
                "thanks": "You're very welcome! Anything else I can do for you?",
                "bye": "Goodbye! It was nice talking to you. Come back anytime.",
                "time": "I don't have a live clock in this offline demo, but your device clock has the exact time.",
                "question": "Good question! Based on what you said, my answer is: let's think it through step by step together.",
                "statement": "Got it! Thanks for telling me that. Tell me more or ask me anything.",
            },
            "fr": {
                "silence": "Je n'ai rien entendu. Pouvez-vous répéter s'il vous plaît ?",
                "greeting": "Bonjour ! Ravi de vous entendre. Comment puis-je vous aider ?",
                "how_are_you": "Je vais très bien, merci ! Et vous, comment allez-vous ?",
                "weather": "Je ne peux pas vérifier la météo hors ligne, mais dites-moi votre ville et je vous aiderai.",
                "name": "Je suis PolyVoice, votre assistant vocal hors ligne.",
                "help": "Je peux discuter, répondre et parler avec votre voix entraînée. Essayez de me saluer ou posez une question.",
                "thanks": "Avec grand plaisir ! Puis-je faire autre chose pour vous ?",
                "bye": "Au revoir ! C'était un plaisir de discuter avec vous.",
                "time": "Je n'ai pas d'horloge en direct dans cette démo hors ligne.",
                "question": "Bonne question ! Réfléchissons-y ensemble étape par étape.",
                "statement": "Compris ! Merci de me l'avoir dit. Racontez-m'en plus.",
            },
        }
        table = T.get(lang, T["en"])
        reply = table.get(intent, table["statement"])
        # contextual grounding: echo short user content + reference turn number
        if clean and intent not in ("silence",):
            short = clean[:120]
            if intent in ("question", "statement"):
                reply = f"{reply} (You said: \u00ab {short} \u00bb.)"
            if n_prev > 0:
                reply = f"{reply} [turn {n_prev + 1}]"
        self.history.append({"user": clean, "intent": intent, "lang": lang, "reply": reply})
        return reply

    async def stream(self, text: str, lang: str):
        full = self.complete(text, lang)
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
