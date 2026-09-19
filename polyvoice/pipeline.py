"""Core streaming pipeline: audio in -> enhanced -> ASR+LID -> brain -> prosody TTS."""
from __future__ import annotations
import asyncio
import re
from dataclasses import dataclass, field

from .audio_io import to_mono_16k, normalize_loudness, denoise_spectral_gate, vad_energy
from .preprocess import preprocess_audio
from .backends import STTBackend, LLMBackend, TTSBackend, DummySTT, DummyLLM, DummyTTS


DISFLU = re.compile(r"\b(uh+|um+|uhm|euh+|hmm+|ah+|er+)\b[,. ]*", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?…؟।。！？])\s+")

EMO_POS = {"happy", "great", "love", "merci", "wonderful", "super", "génial", "bravo"}
EMO_NEG = {"sad", "sorry", "désolé", "angry", "worried", "triste", "peur"}


def clean_transcript(t: str) -> str:
    t = DISFLU.sub(" ", t or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def detect_emotion(text: str) -> str:
    low = text.lower()
    if any(w in low for w in EMO_NEG):
        return "empathetic"
    if any(w in low for w in EMO_POS):
        return "warm"
    if "?" in text or text.strip().endswith(("quoi", "quoi?", "pourquoi")):
        return "curious"
    return "neutral"


def prosody_for(emotion: str, lang: str) -> dict:
    """Superhuman prosody: steady rate, lifted clarity, emotion-tuned pitch,
    shifted by language-family tempo. Humans mumble; we don't. Any ISO code ok."""
    try:
        from .universal import prosody_for_lang
        return prosody_for_lang(emotion, lang)
    except Exception:
        base = {"rate": 1.0, "pitch_hz": "+0Hz", "rate_pct": "+0%", "energy": "medium"}
        if emotion == "warm":
            base.update({"rate": 1.02, "rate_pct": "+2%", "pitch_hz": "+2Hz"})
        elif emotion == "empathetic":
            base.update({"rate": 0.92, "rate_pct": "-8%", "pitch_hz": "-1Hz"})
        elif emotion == "curious":
            base.update({"rate": 1.0, "rate_pct": "+0%", "pitch_hz": "+3Hz"})
        return base


def split_sentences(text: str) -> list[str]:
    parts = SENT_SPLIT.split(text.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # keep TTS chunks small for streaming (<500ms first-byte target)
        while len(p) > 220:
            cut = p.rfind(",", 0, 220)
            if cut < 40:
                cut = 220
            out.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        out.append(p)
    return out or ([text.strip()] if text.strip() else [])


@dataclass
class PipelineConfig:
    sr_in: int = 16000
    auto_lang: bool = True       # reply in user's language
    force_lang: str | None = None
    enhance: bool = True
    first_byte_budget_ms: int = 500


@dataclass
class PolyVoicePipeline:
    stt: STTBackend = field(default_factory=DummySTT)
    llm: LLMBackend = field(default_factory=DummyLLM)
    tts: TTSBackend = field(default_factory=DummyTTS)
    cfg: PipelineConfig = field(default_factory=PipelineConfig)

    # ---- non-streaming one-shot (file in -> wav out) ----
    def run_utterance(self, wav, sr: int) -> dict:
        import numpy as np
        # preprocess BEFORE the model: mono/resample/DC/gentle-denoise/norm/trim
        clean, prep = preprocess_audio(np.asarray(wav), sr, 16000)
        # STT gets the FULL preprocessed utterance: chopping/concatenating
        # VAD fragments destroys sentence timing real recognizers need for LID.
        # (Backends with their own VAD, e.g. faster-whisper vad_filter, handle it.)
        active = clean
        text, lang, conf = self.stt.transcribe(active)
        text = clean_transcript(text)
        if self.cfg.force_lang:
            lang = self.cfg.force_lang
        elif self.cfg.auto_lang is False:
            lang = "en"
        reply = self.llm.complete(text, lang)
        emo = detect_emotion(reply)
        style = prosody_for(emo, lang)
        chunks = split_sentences(reply)
        wavs = [self.tts.synth(c, lang, style)[0] for c in chunks]
        import numpy as _np
        if wavs and hasattr(wavs[0], "shape"):
            out = _np.concatenate([_np.asarray(w, dtype=_np.float32).ravel() for w in wavs])
            sr_out = getattr(self.tts, "sr", 24000)
        else:
            out, sr_out = wavs, getattr(self.tts, "sr", 24000)
        return {"text": text, "lang": lang, "conf": conf, "reply": reply,
                "emotion": emo, "style": style, "audio": out, "sr": sr_out,
                "clean": clean, "prep": prep}

    # ---- streaming: LLM tokens -> sentence TTS chunks ----
    async def stream_utterance(self, wav, sr: int):
        """Yields dicts: {type: asr|llm_tok|tts_chunk|done} for realtime UI."""
        import numpy as np
        clean, _prep = preprocess_audio(np.asarray(wav), sr, 16000)
        text, lang, conf = self.stt.transcribe(clean)
        text = clean_transcript(text)
        if self.cfg.force_lang:
            lang = self.cfg.force_lang
        yield {"type": "asr", "text": text, "lang": lang, "conf": conf}
        buf, full = "", ""
        async for tok in self.llm.stream(text, lang):
            full += tok
            buf += tok
            yield {"type": "llm_tok", "tok": tok}
            sents = split_sentences(buf)
            if len(sents) > 1:  # first complete sentence ready -> synth immediately
                emo = detect_emotion(full)
                style = prosody_for(emo, lang)
                wav_c, sr_c = self.tts.synth(sents[0], lang, style)
                yield {"type": "tts_chunk", "text": sents[0], "audio": wav_c, "sr": sr_c, "style": style}
                buf = " ".join(sents[1:])
        if buf.strip():
            emo = detect_emotion(full)
            style = prosody_for(emo, lang)
            wav_c, sr_c = self.tts.synth(buf.strip(), lang, style)
            yield {"type": "tts_chunk", "text": buf.strip(), "audio": wav_c, "sr": sr_c, "style": style}
        yield {"type": "done", "reply": full, "lang": lang}
