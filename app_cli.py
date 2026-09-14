"""CLI: audio file in -> spoken reply out. Works offline with dummy backends."""
from __future__ import annotations
import argparse
import numpy as np

from polyvoice.pipeline import PolyVoicePipeline, PipelineConfig
from polyvoice.backends import DummySTT, DummyLLM, DummyTTS


def load_audio(path: str) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        x, sr = sf.read(path, always_2d=False)
        return np.asarray(x, dtype=np.float32), int(sr)
    except Exception:
        # fallback: raw 16k mono float32
        x = np.fromfile(path, dtype=np.float32)
        return x, 16000


def main() -> None:
    ap = argparse.ArgumentParser(description="PolyVoice: audio in, better-than-human speech out")
    ap.add_argument("--in", dest="inp", required=True, help="input wav/flac/ogg or raw f32")
    ap.add_argument("--out", dest="out", default="reply.wav", help="output wav path")
    ap.add_argument("--force-lang", default=None, help="e.g. en, fr, es, hi (default: auto same-language)")
    ap.add_argument("--use-real", action="store_true", help="use faster-whisper + edge-tts if installed")
    a = ap.parse_args()

    stt, llm, tts = DummySTT(), DummyLLM(), DummyTTS()
    if a.use_real:
        try:
            from polyvoice.backends import FasterWhisperSTT, EdgeTTSBackend
            stt = FasterWhisperSTT()
            print("[polyvoice] using faster-whisper STT")
        except Exception as e:
            print(f"[polyvoice] real STT unavailable ({e}), using dummy")
    cfg = PipelineConfig(force_lang=a.force_lang)
    pipe = PolyVoicePipeline(stt=stt, llm=llm, tts=tts, cfg=cfg)
    x, sr = load_audio(a.inp)
    r = pipe.run_utterance(x, sr)
    print(f"heard [{r['lang']}@{r['conf']:.2f}]: {r['text']}")
    print(f"reply [{r['emotion']} {r['style']}]: {r['reply']}")
    try:
        import soundfile as sf
        import numpy as _np
        audio = _np.asarray(r["audio"], dtype=_np.float32)
        sf.write(a.out, audio, r["sr"])
        print(f"wrote {a.out} sr={r['sr']} n={len(audio)}")
    except Exception as e:
        print(f"could not write wav ({e}); audio len={len(r['audio']) if hasattr(r['audio'],'__len__') else '?'}")

if __name__ == "__main__":
    main()
