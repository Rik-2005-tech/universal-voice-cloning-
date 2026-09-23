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
class AcousticSTT(STTBackend):
    """True raw-audio path: analyses the waveform itself, no hint needed.

    Uses polyvoice.audio_understand to describe WHAT was heard
    (speech duration, question-like rise, energy, tempo) as a descriptor
    string the ContextLLM answers. Any language, numpy-only, offline.
    `descriptor` may carry a precomputed raw-audio analysis (better than
    re-analysing post-denoise audio inside the pipeline).
    """
    lang: str = "en"
    conf: float = 0.55
    descriptor: str = ""

    def transcribe(self, wav16k, sr: int = 16000):
        import numpy as np
        x = np.asarray(wav16k, dtype=float)
        loud = float((x ** 2).mean() ** 0.5) if len(x) else 0.0
        if loud < 1e-4:
            return "", "en", 0.0
        if self.descriptor:
            return self.descriptor, self.lang, float(self.conf)
        try:
            from .audio_understand import analyze_audio
            info = analyze_audio(wav16k, sr)
        except Exception:
            info = {"speech": False, "descriptor": "[silence]"}
        if not info.get("speech"):
            return "", "en", 0.0
        return info["descriptor"], self.lang, float(self.conf)


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
        if low.startswith("[silence"):
            return "silence"
        if low.startswith("[speech"):
            # acoustic descriptor: question-like rise -> heard_question else heard_statement
            if "question=yes" in low:
                return "heard_question"
            return "heard_statement"
        if any(k in low for k in ("bye", "goodbye", "au revoir", "adios", "tschuss", "alvida", "biday", "अलविदा", "বিদায়")) or low.strip() == "bye":
            return "bye"
        if any(k in low for k in ("thank", "merci", "gracias", "danke", "dhanyavad", "dhonnobad", "shukriya", "shukria", "धन्यवाद", "ধন্যবাদ")):
            return "thanks"
        if any(k in low for k in ("weather", "météo", "clima", "wetter", "mausam", "abohawa", "मौसम", "আবহাওয়া")):
            return "weather"
        if any(k in low for k in ("your name", "who are you", "comment tu t'appelles", "qui es-tu",
                                  "cómo te llamas", "wie heißt du", "tumhara naam", "aapka naam",
                                  "tumar nam", "tumi ke", "aapka naam kya", "नाम क्या", "নাম কী")):
            return "name"
        if any(k in low for k in ("help", "aide", "ayuda", "hilfe", "madad", "sahajjo", "sahayata", "मदद", "सहायता", "সাহায্য")):
            return "help"
        if any(k in low for k in ("how are you", "how is it going", "comment vas-tu", "comment ça va",
                                  "cómo estás", "wie geht", "kaise ho", "how are u",
                                  "kemon acho", "kemon achen", "aap kaise",
                                  "कैसे हैं", "कैसे हो", "কেমন আছেন", "কেমন আছো")):
            return "how_are_you"
        if any(k in low for k in ("hello", "hi", "hey", "bonjour", "salut", "hola",
                                  "hallo", "namaste", "namaskar", "नमस्ते", "নমস্কার",
                                  "good morning", "good afternoon")):
            return "greeting"
        if any(k in low for k in ("time", "heure", "hora", "uhr", "samay", "kitne baje", "somoy", "koyta", "समय", "সময়", "কয়টা")):
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
                "heard_question": "I hear you asking something! I can't make out the exact words offline, but ask me again with a text hint and I'll answer fully.",
                "heard_statement": "I hear you loud and clear! I can't make out the exact words offline — send the words as text and I'll reply in full.",
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
                "heard_question": "Je vous entends poser une question ! Je ne distingue pas les mots exacts hors ligne — envoyez le texte et je répondrai.",
                "heard_statement": "Je vous entends très bien ! Je ne distingue pas les mots exacts hors ligne — envoyez le texte et je répondrai.",
            },
            "es": {
                "silence": "No escuché nada. ¿Puedes hablar de nuevo, por favor?",
                "greeting": "¡Hola! Qué bueno escucharte. ¿Cómo puedo ayudarte hoy?",
                "how_are_you": "¡Estoy muy bien, gracias! ¿Y tú, cómo estás?",
                "weather": "No puedo ver el clima sin conexión, pero dime tu ciudad y te ayudo.",
                "name": "Soy PolyVoice, tu asistente de voz sin conexión.",
                "help": "Puedo charlar, responder y hablar con tu voz entrenada. Salúdame o haz una pregunta.",
                "thanks": "¡Con mucho gusto! ¿Algo más en lo que pueda ayudar?",
                "bye": "¡Adiós! Fue un placer hablar contigo.",
                "time": "No tengo reloj en vivo en esta demo sin conexión.",
                "question": "¡Buena pregunta! Pensémoslo juntos paso a paso.",
                "statement": "¡Entendido! Gracias por contarme. Cuéntame más.",
                "heard_question": "¡Te escucho preguntar algo! No distingo las palabras exactas sin conexión — envía el texto y responderé.",
                "heard_statement": "¡Te escucho alto y claro! No distingo las palabras exactas sin conexión — envía el texto y responderé.",
            },
            "hi": {
                "silence": "Maine kuch nahi suna. Kripya phir se bolein?",
                "greeting": "Namaste! Aapko sunkar achha laga. Main aapki kaise madad kar sakta hun?",
                "how_are_you": "Main bahut achha hun, dhanyavad! Aap kaise hain?",
                "weather": "Main offline mausam nahi dekh sakta, apna sheher batayein.",
                "name": "Main PolyVoice hun, aapka offline voice assistant.",
                "help": "Main baat kar sakta hun, jawab de sakta hun. Namaste kahein ya sawal poochhein.",
                "thanks": "Bahut khushi hui! Kya main aur kuch kar sakta hun?",
                "bye": "Alvida! Aapse baat karke achha laga.",
                "time": "Is offline demo mein live ghadi nahi hai.",
                "question": "Achha sawal! Chaliye ise milkar suljhate hain.",
                "statement": "Samajh gaya! Batane ke liye dhanyavad. Aur batayein.",
                "heard_question": "Main sun raha hun ki aap kuch poochh rahe hain! Shabd saaf nahi hain — text bhejein, main jawab dunga.",
                "heard_statement": "Main aapko sun raha hun! Shabd saaf nahi hain — text bhejein, main jawab dunga.",
            },
            "zh": {
                "silence": "我什么都没听到，请再说一遍好吗？",
                "greeting": "你好！很高兴听到你。今天我能帮你什么？",
                "how_are_you": "我很好，谢谢！你怎么样？",
                "weather": "离线时我看不到天气，请告诉我你的城市。",
                "name": "我是PolyVoice，你的离线语音助手。",
                "help": "我可以聊天、回答问题。请打招呼或提问。",
                "thanks": "不客气！还有什么可以帮你吗？",
                "bye": "再见！和你聊天很愉快。",
                "time": "这个离线演示没有实时时钟。",
                "question": "好问题！我们一起一步一步想想。",
                "statement": "明白了！谢谢你告诉我。再多说一点吧。",
                "heard_question": "我听到你在问问题！离线时听不清每个字——把文字发给我，我会完整回答。",
                "heard_statement": "我清楚地听到你了！离线时听不清每个字——把文字发给我，我会完整回答。",
            },
            "ar": {
                "silence": "لم أسمع شيئا. هل يمكنك التحدث مرة أخرى؟",
                "greeting": "مرحبا! سعيد بسماعك. كيف يمكنني مساعدتك اليوم؟",
                "how_are_you": "أنا بخير جدا، شكرا! وكيف حالك؟",
                "weather": "لا أستطيع رؤية الطقس دون اتصال، أخبرني بمدينتك.",
                "name": "أنا PolyVoice، مساعدك الصوتي دون اتصال.",
                "help": "يمكنني الدردشة والإجابة. حيّني أو اطرح سؤالا.",
                "thanks": "على الرحب والسعة! هل يمكنني فعل شيء آخر؟",
                "bye": "وداعا! سعدت بالحديث معك.",
                "time": "لا توجد ساعة مباشرة في هذه النسخة دون اتصال.",
                "question": "سؤال جيد! لنفكر فيه معا خطوة بخطوة.",
                "statement": "فهمت! شكرا لإخباري. حدثني أكثر.",
                "heard_question": "أسمعك تطرح سؤالا! لا أميز الكلمات بدقة دون اتصال — أرسل النص وسأجيب بالكامل.",
                "heard_statement": "أسمعك بوضوح! لا أميز الكلمات بدقة دون اتصال — أرسل النص وسأجيب بالكامل.",
            },
            "bn": {
                "silence": "আমি কিছু শুনতে পাইনি। দয়া করে আবার বলুন?",
                "greeting": "নমস্কার! আপনাকে শুনে ভালো লাগলো। আজ আমি কীভাবে সাহায্য করতে পারি?",
                "how_are_you": "আমি খুব ভালো আছি, ধন্যবাদ! আপনি কেমন আছেন?",
                "weather": "অফলাইনে আমি আবহাওয়া দেখতে পাই না, আপনার শহরের নাম বলুন।",
                "name": "আমি PolyVoice, আপনার অফলাইন ভয়েস সহকারী।",
                "help": "আমি কথা বলতে পারি, উত্তর দিতে পারি। নমস্কার বলুন বা প্রশ্ন করুন।",
                "thanks": "অনেক স্বাগত! আর কিছু করতে পারি কি?",
                "bye": "বিদায়! আপনার সাথে কথা বলে ভালো লাগলো।",
                "time": "এই অফলাইন ডেমোতে লাইভ ঘড়ি নেই।",
                "question": "ভালো প্রশ্ন! চলুন একসাথে ধাপে ধাপে ভাবি।",
                "statement": "বুঝেছি! জানানোর জন্য ধন্যবাদ। আরও বলুন।",
                "heard_question": "শুনছি আপনি কিছু জিজ্ঞেস করছেন! অফলাইনে প্রতিটি শব্দ বুঝতে পারছি না — লিখে পাঠান, পুরো উত্তর দেব।",
                "heard_statement": "আপনাকে স্পষ্ট শুনতে পাচ্ছি! অফলাইনে প্রতিটি শব্দ বুঝতে পারছি না — লিখে পাঠান, পুরো উত্তর দেব।",
            },
        }
        table = T.get(lang, T["en"])
        reply = table.get(intent, table["statement"])
        # contextual grounding: echo short user content + reference turn number
        # (skip echo for raw-audio descriptors and silence)
        if clean and intent not in ("silence", "heard_question", "heard_statement"):
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
    micro: str | None = None
    _micro_bank: object = None
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
        import copy as _c
        from .universal import synth_universal, adapt_to_lang, prior_for, normalize_lang
        from .humanize import humanize
        micro_res = None
        if self.micro:
            try:
                from .timbre import load_micro_bank, apply_eq
                if self._micro_bank is None:
                    self._micro_bank = load_micro_bank(self.micro)
                per, aux = self._micro_bank
                bank, glob = per, aux.get("glob", None)
                res = aux.get("res", {})
                micro_res = res.get(normalize_lang(lang), res.get("und"))
                _apply_eq = apply_eq
            except Exception:
                micro_res = None
                bank, glob = self._bank()
        else:
            bank, glob = self._bank()
        # adapt_to_lang returns a guarded copy (human pitch range)
        p = _c.copy(adapt_to_lang(lang, bank, glob))
        if style and "rate" in style:
            p.rate = float(style["rate"])
        wav = synth_universal(text, lang, p, self.sr)
        f0ref = prior_for(lang)["f0"]
        wav = humanize(wav, self.sr, text, f0=f0ref)
        if micro_res is not None:
            try:
                wav = _apply_eq(wav, self.sr, micro_res)
            except Exception:
                pass
        return wav, self.sr


@dataclass
class PiperTTS(TTSBackend):
    """Real talking voice: neural Piper TTS speaks intelligible words with a
    human voice (offline, after one ~60MB download). English uses Piper;
    other languages fall back to the trained SuperhumanTTS hum. If the model
    file is missing, everything falls back gracefully (never crashes)."""
    model: str = "voices/en-lessac-medium.onnx"
    fallback_ckpt: str = "bank_large.npz"
    sr: int = 22050
    _v: object = None
    _fb: object = None
    _voices: object = None

    VOICE_MAP = {
        "en": "voices/en-lessac-medium.onnx",
        "hi": "voices/hi-pratham-medium.onnx",
        "bn": "voices/bn-google-medium.onnx",
        "fr": "voices/fr-siwis-medium.onnx",
        "es": "voices/es-davefx-medium.onnx",
        "zh": "voices/zh-huayan-medium.onnx",
        "ar": "voices/ar-kareem-medium.onnx",
    }

    def _load(self):
        return self._load_voice(self.model)

    def _load_voice(self, path: str):
        if self._voices is None:
            self._voices = {}
        if path not in self._voices:
            from piper import PiperVoice
            import os as _os
            if not _os.path.exists(path):
                raise FileNotFoundError(f"piper voice missing: {path}")
            self._voices[path] = PiperVoice.load(path)
        v = self._voices[path]
        try:
            self.sr = int(v.config.sample_rate)
        except Exception:
            pass
        return v

    def synth(self, text: str, lang: str, style: dict | None = None):
        try:
            from .universal import normalize_lang
        except Exception:
            normalize_lang = lambda l: (l or "en")  # noqa
        path = self.VOICE_MAP.get(normalize_lang(lang), self.model)
        try:
            v = self._load_voice(path)
        except Exception:
            if self._fb is None:
                self._fb = SuperhumanTTS(ckpt=self.fallback_ckpt)
            return self._fb.synth(text, lang, style)
        import io
        import wave
        import numpy as _np
        rate = 1.0
        try:
            rate = float((style or {}).get("rate", 1.0))
        except Exception:
            rate = 1.0
        length_scale = float(_np.clip(1.0 / max(0.5, rate), 0.6, 1.5))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            try:
                from piper.config import SynthesisConfig
                cfg = SynthesisConfig(length_scale=length_scale)
            except Exception:
                cfg = None
            v.synthesize_wav(str(text), w, syn_config=cfg)
        buf.seek(0)
        with wave.open(buf, "rb") as r:
            raw = r.readframes(r.getnframes())
            out_sr = r.getframerate()
        audio = (_np.frombuffer(raw, dtype=_np.int16).astype(_np.float32) / 32768.0)
        return audio, out_sr


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

# Expected native script per language (Unicode blocks). Used to catch LID
# votes that contradict the transcript actually produced.
_LANG_SCRIPTS = {
    "bn": [("0980", "09FF")],   # Bengali
    "hi": [("0900", "097F")],   # Devanagari
    "zh": [("4E00", "9FFF"), ("3040", "30FF"), ("AC00", "D7FF")],  # CJK
    "yue": [("4E00", "9FFF")],
    "ar": [("0600", "06FF")],   # Arabic
    "ur": [("0600", "06FF")],
    "fa": [("0600", "06FF")],
    "he": [("0590", "05FF")],   # Hebrew
    "ru": [("0400", "04FF")],   # Cyrillic
    "uk": [("0400", "04FF")],
    "el": [("0370", "03FF")],   # Greek
    "th": [("0E00", "0E7F")],   # Thai
    "ta": [("0B80", "0BFF")],   # Tamil
    "te": [("0C00", "0C7F")],   # Telugu
    "kn": [("0C80", "0CFF")],   # Kannada
    "ml": [("0D00", "0D7F")],   # Malayalam
}

# Romanized keywords proving the transcript really is that language
# (whisper often romanizes Hindi/Bengali instead of native script).
_ROMANIZED = {
    "bn": ("nomoshkar", "namaskar", "kemon", "kamon", "achen", "acho", "dhonnobad",
           "bhalo", "biday", "tumi", "apni", "apnar", "ki", "kothai", "akhon"),
    "hi": ("namaste", "kaise", "dhanyavad", "shukriya", "alvida", "aapka",
           "tumhara", "madad", "mausam", "kya", "kaise"),
}

_EN_COMMON = ("the", "is", "are", "what", "how", "and", "you", "that", "this",
              "with", "have", "from", "page", "test", "your", "about", "more")


def _script_consistent(text: str, lang: str) -> bool:
    """True if the transcript plausibly matches the detected language."""
    import re as _re
    low = (text or "").lower()
    words = _re.findall(r"[a-z']+", low)
    if not words:
        return True  # native script only (or empty) — trust the LID vote
    ranges = _LANG_SCRIPTS.get((lang or "").split("-")[0].lower())
    if ranges and any(any(int(a, 16) <= ord(c) <= int(b, 16) for a, b in ranges)
                      for c in (text or "")):
        return True  # native script present — trust the LID vote
    if ranges:
        # romanized? look for that language's keywords
        keys = _ROMANIZED.get((lang or "").split("-")[0].lower(), ())
        if keys and any(k in low for k in keys):
            return True
        # pure English sentences mislabelled: flip to English
        en_hits = sum(1 for w in words if w in _EN_COMMON)
        if en_hits >= 3 and len(words) >= 6:
            return False
    return True


class FasterWhisperSTT(STTBackend):
    """Streaming-capable multilingual ASR with built-in LID. Requires faster-whisper."""
    def __init__(self, model: str = "small", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(model, device=device, compute_type=compute_type)
    def transcribe(self, wav16k, sr: int = 16000):
        import numpy as np
        x0 = np.asarray(wav16k, dtype=float)
        if len(x0) == 0 or float((x0 ** 2).mean() ** 0.5) < 1e-4:
            return "", "en", 0.0
        x = np.asarray(wav16k, dtype=np.float32)
        segs, info = self.m.transcribe(x, beam_size=1, vad_filter=True)
        txt = " ".join(s.text.strip() for s in segs).strip()
        lang = (info.language or "en")
        conf = float(info.language_probability or 0.0)
        if txt and not _script_consistent(txt, lang):
            lang, conf = "en", conf * 0.8
        return txt, lang, conf


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
