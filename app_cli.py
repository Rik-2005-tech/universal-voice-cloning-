"""CLI: audio file in -> spoken reply out. Works offline with trained voice."""
from __future__ import annotations
import argparse
import os
import numpy as np

from polyvoice.pipeline import PolyVoicePipeline, PipelineConfig
from polyvoice.backends import TranscriptSTT, ContextLLM, SuperhumanTTS, DummyTTS


def load_audio(path: str) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        x, sr = sf.read(path, always_2d=False)
        return np.asarray(x, dtype=np.float32), int(sr)
    except Exception:
        # fallback: raw 16k mono float32
        x = np.fromfile(path, dtype=np.float32)
        return x, 16000


def resolve_transcript(inp_path: str, cli_text: str | None, cli_lang: str | None) -> tuple[str, str]:
    """Priority: --text > <inp>.txt sidecar > default English."""
    if cli_text:
        return cli_text, (cli_lang or "en")
    sidecar = os.path.splitext(inp_path)[0] + ".txt"
    if os.path.exists(sidecar):
        try:
            with open(sidecar, encoding="utf-8") as fh:
                t = fh.read().strip()
            if t:
                return t, (cli_lang or "en")
        except Exception:
            pass
    return "Hello, how are you today?", (cli_lang or "en")


def main() -> None:
    ap = argparse.ArgumentParser(description="PolyVoice: audio in, contextual spoken reply out")
    ap.add_argument("--in", dest="inp", required=True, help="input wav/flac/ogg or raw f32")
    ap.add_argument("--out", dest="out", default="reply.wav", help="output wav path")
    ap.add_argument("--text", default=None, help="transcript override (simulates ASR without model download)")
    ap.add_argument("--lang", default=None, help="input language code (default: en; auto from --text/sidecar)")
    ap.add_argument("--force-lang", default=None, help="reply language (default: same as input)")
    ap.add_argument("--ckpt", default=None, help="trained voice bank (default: bank_multi.npz > bank_en.npz)")
    ap.add_argument("--use-real", action="store_true", help="use faster-whisper if installed")
    a = ap.parse_args()

    text, lang = resolve_transcript(a.inp, a.text, a.lang)
    stt = TranscriptSTT(text=text, lang=lang)
    if a.use_real:
        try:
            from polyvoice.backends import FasterWhisperSTT
            stt = FasterWhisperSTT()
            print("[polyvoice] using faster-whisper STT")
        except Exception as e:
            print(f"[polyvoice] real STT unavailable ({e}), using offline TranscriptSTT")

    llm = ContextLLM()
    ckpt = a.ckpt
    if ckpt is None:
        for cand in ("bank_large.npz", "bank_multi.npz", "bank_en.npz", "bank.npz"):
            if os.path.exists(cand):
                ckpt = cand
                break
    if ckpt and os.path.exists(ckpt):
        tts = SuperhumanTTS(ckpt=ckpt)
        print(f"[polyvoice] using trained voice {ckpt}")
    else:
        tts = DummyTTS()
        print(f"[polyvoice] no bank found ({ckpt}), using DummyTTS")

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
