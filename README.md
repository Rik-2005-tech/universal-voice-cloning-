# PolyVoice — offline multilingual talking assistant

Audio in → preprocess → understand (any language) → contextual reply → spoken voice out.
Hindi, Bengali and English are first-class; French, Spanish, Chinese, Arabic included;
unseen languages fall back gracefully instead of crashing.

## Quick start

```bash
pip install -r requirements.txt          # numpy, soundfile, piper-tts, faster-whisper
python3 app_cli.py --in test/input/mytest.wav --out test/output/mytest_reply.wav
```

One input produces four outputs in `test/output/`:

| File | Contents |
|---|---|
| `mytest_reply.wav` | spoken reply (real human voice) |
| `mytest_reply.clean.wav` | what the model actually heard (preprocessed 16kHz) |
| `mytest_reply.analysis.txt` | detected language, confidence, acoustic features, heard text |
| `mytest_reply.reply.txt` | emotion, reply language, reply text |

Transcript priority: `--text "..."` > `<input>.txt` sidecar > faster-whisper
(auto language detect) > acoustic analysis (tone/shape, any language, zero hints).

## How it works

```
audio.wav → preprocess → STT/LID → ContextLLM → TTS → reply.wav
              (mono/16k/DC/        (faster-whisper   (intent +      (Piper neural
               light-denoise/       auto, 100         memory,        voices per
               norm/trim)            langs)            7 langs)        language)
```

* **Ears** — `polyvoice/preprocess.py` (clean model-ready audio) and
  `polyvoice/audio_understand.py` (speech, question-like rise, energy, tempo).
* **Brain** — `polyvoice/backends.py:ContextLLM`: greeting/how-are-you/weather/
  name/help/thanks/bye/time/question/statement + silence/acoustic fallbacks,
  per-session memory, replies in en/hi/bn/fr/es/zh/ar.
* **Mouth** — `polyvoice/backends.py:PiperTTS`: offline neural voices
  (`voices/`, ~60MB each, downloaded once) with trained-voice fallback.
* **Trained voice** — `polyvoice/universal.py` (any-language adapters, zero-shot
  blends, pitch guard) + `polyvoice/timbre.py` (spectral fingerprint copied
  from real voices, applied as EQ).

## Training

| Script | Data | Output |
|---|---|---|
| `train_voice.py` | 64 synthetic utterances, 7 langs | `bank_multi.npz` (now incl. Bengali) |
| `train_large.py` | 1,440 mixed: templates + Tagore/Premchand/Austen novels (`corpora/`, `import_novels.py`) | `bank_large.npz` |
| `train_micro.py` | **3,600 real clips**: GramVaani Hindi, LibriSpeech English, SLR37 Bengali (`data_voice/`, see `fetch_voice_data.py`, `convert_*.py`) | `bank_micro.npz` (params + per-lang EQ residuals + pitch stats) |

Real measured voice stats: Bengali 152.8Hz/wobble 0.19, English 175Hz/0.21,
Hindi 224Hz/0.31. Voice-color distance to real voices after micro-EQ:
Hindi **44% closer**, Bengali **19% closer**. `pytest`: 7 passed.

Large files (`voices/`, `data_voice/`) are intentionally **not** in git —
re-download with `check_voices.py` deps + `fetch_voice_data.py` /
`convert_librispeech.py` / `convert_hindi.py` (+ SLR37 `bn_in.zip`).

## Honest limits

* Exact words come from Piper's built-in speakers, not a clone of your voice
  (true cloning needs GPU models like XTTS — see `colab_*` helpers).
* Short Bengali clips transcribe transliterated at low confidence (language ID
  itself is reliable); use `--text` for word-exact answers.
* The sine-synth fallback hums prosody; it cannot form words.
