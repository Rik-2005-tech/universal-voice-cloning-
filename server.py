"""FastAPI websocket server for realtime speech-to-speech.
Run: uvicorn server:app --host 0.0.0.0 --port 8000
Protocol: client sends binary PCM16 16k mono chunks; server streams back
JSON {asr/llm_tok} + binary reply audio chunks. First audio <500ms after endpoint.
"""
from __future__ import annotations

try:
    from fastapi import FastAPI, WebSocket
    app = FastAPI(title="PolyVoice streaming")
    HAVE = True
except Exception:
    app = None
    HAVE = False

import asyncio
import numpy as np
from polyvoice.pipeline import PolyVoicePipeline
from polyvoice.backends import DummySTT, DummyLLM, DummyTTS, ContextLLM


def build_pipeline() -> PolyVoicePipeline:
    """Fresh pipeline per connection: stateless STT/TTS may be shared, but
    the brain (conversation memory) MUST be per-session, otherwise users
    would see each other's turns."""
    return PolyVoicePipeline(stt=DummySTT(), llm=ContextLLM(), tts=DummyTTS())

if HAVE:
    @app.get("/health")
    def health():
        return {"ok": True}

    @app.websocket("/ws/speak")
    async def ws_speak(ws: WebSocket):
        await ws.accept()
        pipe = build_pipeline()  # per-connection brain: no cross-user leaks
        buf = bytearray()
        try:
            while True:
                msg = await ws.receive()
                if msg.get("bytes") is not None:
                    buf.extend(msg["bytes"])
                    # endpoint heuristic: 1.2s buffered -> decode utterance
                    if len(buf) >= 16000 * 2 * 1.2:
                        wav = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0
                        buf.clear()
                        async for ev in pipe.stream_utterance(wav, 16000):
                            if ev["type"] == "tts_chunk":
                                a = np.asarray(ev["audio"], dtype=np.float32)
                                await ws.send_bytes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
                            else:
                                await ws.send_json({k: v for k, v in ev.items() if k not in ("audio",)})
                elif msg.get("text") == "__flush__":
                    if buf:
                        wav = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0
                        buf.clear()
                        async for ev in pipe.stream_utterance(wav, 16000):
                            if ev["type"] == "tts_chunk":
                                a = np.asarray(ev["audio"], dtype=np.float32)
                                await ws.send_bytes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
                            else:
                                await ws.send_json({k: v for k, v in ev.items() if k not in ("audio",)})
                    await ws.send_json({"type": "turn_end"})
        except Exception as e:
            try:
                await ws.send_json({"type": "error", "msg": str(e)})
                await ws.close()
            except Exception:
                pass
