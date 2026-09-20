"""CLI: audio file in -> spoken reply out. Works offline with trained voice."""
from __future__ import annotations
import argparse
import os
import numpy as np

from polyvoice.pipeline import PolyVoicePipeline, PipelineConfig
from polyvoice.backends import TranscriptSTT, AcousticSTT, ContextLLM, SuperhumanTTS, DummyTTS, PiperTTS
from polyvoice.audio_understand import analyze_audio


def load_audio(path: str) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        x, sr = sf.read(path, always_2d=False)
        return np.asarray(x, dtype=np.float32), int(sr)
    except Exception:
        # fallback: raw 16k mono float32
        x = np.fromfile(path, dtype=np.float32)
        return x, 16000


def resolve_stt(inp_path: str, cli_text: str | None, cli_lang: str | None,
                audio: np.ndarray, sr: int, fw_model: str | None):
    """Audio file in -> STT backend out. Priority:
    --text > <inp>.txt sidecar > faster-whisper (auto language detect +
    transcription, needs install) > raw acoustic analysis of the file itself.
    Returns (stt_backend, analysis_dict)."""
    if cli_text:
        return TranscriptSTT(text=cli_text, lang=(cli_lang or "en")), {"mode": "cli-text"}
    sidecar = os.path.splitext(inp_path)[0] + ".txt"
    if os.path.exists(sidecar):
        try:
            with open(sidecar, encoding="utf-8") as fh:
                t = fh.read().strip()
            if t:
                return TranscriptSTT(text=t, lang=(cli_lang or "en")), {"mode": "sidecar"}
        except Exception:
            pass
    if fw_model:
        try:
            from polyvoice.backends import FasterWhisperSTT
            stt = FasterWhisperSTT(model=fw_model)
            return stt, {"mode": f"auto-stt({fw_model})"}
        except Exception as e:
            print(f"[polyvoice] auto STT unavailable ({e}), using acoustic analysis")
    info = analyze_audio(audio, sr)
    info["mode"] = "acoustic"
    stt = AcousticSTT(lang=(cli_lang or "en"),
                      descriptor=info["descriptor"] if info.get("speech") else "")
    return stt, info


def main() -> None:
    ap = argparse.ArgumentParser(description="PolyVoice: audio in, contextual spoken reply out")
    ap.add_argument("--in", dest="inp", required=True, help="input wav/flac/ogg or raw f32")
    ap.add_argument("--out", dest="out", default="reply.wav", help="output wav path")
    ap.add_argument("--text", default=None, help="transcript override (simulates ASR without model download)")
    ap.add_argument("--lang", default=None, help="input language code (default: en; auto from --text/sidecar)")
    ap.add_argument("--force-lang", default=None, help="reply language (default: same as input)")
    ap.add_argument("--ckpt", default=None, help="trained voice bank (default: bank_large.npz > bank_multi > bank_en)")
    ap.add_argument("--use-real", action="store_true", help="deprecated: auto-STT is now default (see --fw-model)")
    ap.add_argument("--fw-model", default="base", help="faster-whisper model for auto language detect (tiny/base/small; empty to disable)")
    ap.add_argument("--no-auto-stt", action="store_true", help="skip auto speech recognition, use acoustic analysis only")
    ap.add_argument("--analysis-out", default=None, help="analysed-text file (default: <out>.analysis.txt)")
    ap.add_argument("--reply-out", default=None, help="reply-text file (default: <out>.reply.txt)")
    ap.add_argument("--clean-out", default=None, help="preprocessed-audio file (default: <out>.clean.wav)")
    ap.add_argument("--no-clean", action="store_true", help="skip saving preprocessed audio")
    ap.add_argument("--detect", action="store_true", help="print HUMAN-vs-AI voice verdict for the input (fast model)")
    ap.add_argument("--deep", action="store_true", help="use deep neural verdict (slower, ~1min, catches neural voices in en/hi/bn/fr/zh)")
    a = ap.parse_args()

    x, sr = load_audio(a.inp)
    if a.detect or a.deep:
        if a.deep:
            try:
                from polyvoice.deep_spoof import verdict_deep
                v = verdict_deep(x, sr)
                print(f"[polyvoice] deep verdict: {v['label']} "
                      f"(p_fake={v.get('p_fake')}, thr={v.get('threshold')})")
            except Exception as e:
                print(f"[polyvoice] deep verdict unavailable ({e})")
        else:
            try:
                from polyvoice.spoof import verdict
                v = verdict(x, sr)
                print(f"[polyvoice] voice verdict: {v['label']} (p_ai={v.get('p_ai')}, model_acc={v.get('model_acc')})")
            except Exception as e:
                print(f"[polyvoice] verdict unavailable ({e})")
    fw = None if (a.no_auto_stt or (a.fw_model or "").strip() == "") else (a.fw_model or "tiny")
    stt, info = resolve_stt(a.inp, a.text, a.lang, x, sr, fw)
    if info.get("mode") == "acoustic":
        print(f"[polyvoice] analysing raw audio: dur={info.get('dur_s')}s "
              f"question-like={'yes' if info.get('question_like') else 'no'} "
              f"energy={info.get('energy')} tempo={info.get('tempo')}")
    else:
        print(f"[polyvoice] input mode: {info.get('mode')}")

    llm = ContextLLM()
    ckpt = a.ckpt
    if ckpt is None:
        for cand in ("bank_large.npz", "bank_multi.npz", "bank_en.npz", "bank.npz"):
            if os.path.exists(cand):
                ckpt = cand
                break
    # real talking voice first (intelligible words), trained hum as fallback
    if os.path.exists("voices/en-lessac-medium.onnx"):
        tts = PiperTTS(model="voices/en-lessac-medium.onnx",
                       fallback_ckpt=ckpt or "bank_large.npz")
        print("[polyvoice] using Piper talking voice (+ trained fallback)")
    elif ckpt and os.path.exists(ckpt):
        tts = SuperhumanTTS(ckpt=ckpt)
        print(f"[polyvoice] using trained voice {ckpt}")
    else:
        tts = DummyTTS()
        print(f"[polyvoice] no bank found ({ckpt}), using DummyTTS")

    cfg = PipelineConfig(force_lang=a.force_lang)
    pipe = PolyVoicePipeline(stt=stt, llm=llm, tts=tts, cfg=cfg)
    r = pipe.run_utterance(x, sr)
    if a.lang and info.get("mode", "").startswith("auto-stt"):
        # user knows the language better than the recognizer: keep its words,
        # trust the user's language for understanding + reply voice.
        from polyvoice.pipeline import detect_emotion, prosody_for, split_sentences
        print(f"[polyvoice] language override: {r['lang']} -> {a.lang}")
        r["lang"] = a.lang
        r["reply"] = pipe.llm.complete(r["text"], a.lang)
        r["emotion"] = detect_emotion(r["reply"])
        r["style"] = prosody_for(r["emotion"], a.lang)
        parts = [pipe.tts.synth(c, a.lang, r["style"])[0]
                 for c in split_sentences(r["reply"])]
        import numpy as _np3
        r["audio"] = _np3.concatenate(
            [_np3.asarray(w, dtype=_np3.float32).ravel() for w in parts])
        r["sr"] = getattr(pipe.tts, "sr", 24000)
    prep = r.get("prep", {})
    print(f"[polyvoice] preprocessed: {' > '.join(prep.get('steps', []))} "
          f"(dur {prep.get('dur_in_s')}s -> {prep.get('dur_out_s')}s)")
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

    # ---- analysed text + reply text files (beside the voice file) ----
    base, _ = os.path.splitext(a.out)
    analysis_path = a.analysis_out or (base + ".analysis.txt")
    reply_path = a.reply_out or (base + ".reply.txt")
    clean_path = a.clean_out or (base + ".clean.wav")
    if not a.no_clean:
        try:
            clean = _np.asarray(r.get("clean"), dtype=_np.float32)
            sf.write(clean_path, clean, 16000)
            print(f"wrote {clean_path} (model-ready 16kHz input)")
        except Exception as e:
            print(f"could not write clean file ({e})")
    try:
        with open(analysis_path, "w", encoding="utf-8") as fh:
            fh.write(f"input: {a.inp}\n")
            fh.write(f"input_mode: {info.get('mode')}\n")
            fh.write(f"detected_lang: {r['lang']}\n")
            fh.write(f"confidence: {r['conf']:.2f}\n")
            if info.get("mode") == "acoustic":
                for k in ("dur_s", "speech_ratio", "rms", "f0_mean",
                          "question_like", "expressive", "tempo", "energy", "segments"):
                    if k in info:
                        fh.write(f"acoustic_{k}: {info[k]}\n")
            fh.write(f"heard_text: {r['text']}\n")
        with open(reply_path, "w", encoding="utf-8") as fh:
            fh.write(f"emotion: {r['emotion']}\n")
            fh.write(f"reply_lang: {r['lang']}\n")
            fh.write(f"reply_text: {r['reply']}\n")
        print(f"wrote {analysis_path} + {reply_path}")
    except Exception as e:
        print(f"could not write text files ({e})")

if __name__ == "__main__":
    main()
