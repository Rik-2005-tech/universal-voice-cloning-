"""Offline tests: no model download, no mic needed."""
import asyncio
import numpy as np
from polyvoice.pipeline import PolyVoicePipeline, clean_transcript, prosody_for, split_sentences
from polyvoice.audio_io import normalize_loudness, vad_energy
from polyvoice.backends import DummySTT, DummyLLM, DummyTTS


def test_enhance_and_vad():
    sr = 16000
    t = np.arange(sr) / sr
    wav = (0.02 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav[sr // 2:] += (0.2 * np.sin(2 * np.pi * 220 * t[:sr // 2])).astype(np.float32)
    y = normalize_loudness(wav)
    assert abs(float((y ** 2).mean() ** 0.5) - 0.12) < 0.03, "loudness normalize failed"
    segs = vad_energy(y, sr, thresh=0.015)
    assert segs, "VAD should find speech segment"


def test_clean_and_prosody():
    assert clean_transcript("uh hello um world") == "hello world"
    assert prosody_for("empathetic", "fr")["rate"] < 1.0
    assert len(split_sentences("Bonjour ! Comment vas-tu ? Je vais bien.")) >= 2


def test_same_language_routing():
    pipe = PolyVoicePipeline(stt=DummySTT(), llm=DummyLLM(), tts=DummyTTS())
    sr = 16000
    wav = (0.3 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
    r = pipe.run_utterance(wav, sr)
    assert r["lang"] == "fr", f"expected auto fr, got {r['lang']}"
    assert "Bonjour" in r["reply"], "reply should be in user's language (fr)"
    assert hasattr(r["audio"], "shape") and len(r["audio"]) > 1000


def test_streaming_order():
    async def go():
        pipe = PolyVoicePipeline(stt=DummySTT(), llm=DummyLLM(), tts=DummyTTS())
        sr = 16000
        wav = (0.3 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
        types = []
        async for ev in pipe.stream_utterance(wav, sr):
            types.append(ev["type"])
        assert types[0] == "asr" and types[-1] == "done"
        assert "tts_chunk" in types, "streaming must yield audio before done (low latency)"
    asyncio.run(go())

if __name__ == "__main__":
    for f in (test_enhance_and_vad, test_clean_and_prosody, test_same_language_routing, test_streaming_order):
        f()
        print(f"PASS {f.__name__}")
