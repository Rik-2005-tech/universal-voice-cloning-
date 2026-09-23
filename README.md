# PolyVoice — offline multilingual talking assistant + human-vs-AI voice detector

Audio in → preprocess → understand (any language) → contextual reply → spoken voice out.
Hindi, Bengali and English are first-class; French, Spanish, Chinese, Arabic included;
unseen languages fall back gracefully instead of crashing.

## Quick start

```bash
pip install -r requirements.txt          # numpy, soundfile, piper-tts, faster-whisper (+torch for --deep)
python3 app_cli.py --in test/input/mytest.wav --out test/output/mytest_reply.wav
```

One input produces up to five outputs in `test/output/`:

| File | Contents |
|---|---|
| `mytest_reply.wav` | spoken reply (real human voice) |
| `mytest_reply.clean.wav` | what the model actually heard (preprocessed 16kHz) |
| `mytest_reply.analysis.txt` | detected language, confidence, acoustic features, heard text |
| `mytest_reply.reply.txt` | emotion, reply language, reply text |

Transcript priority: `--text "..."` > `<input>.txt` sidecar > faster-whisper
(auto language detect) > acoustic analysis (tone/shape, any language, zero hints).

Voice check on any file (language auto-detected, per-language model called):

```bash
python3 app_cli.py --in test/input/mytest.wav --out test/output/x.wav --detect   # fast (<1s)
python3 app_cli.py --in test/input/mytest.wav --out test/output/x.wav --cascade  # fast + auto deep confirm
python3 app_cli.py --in test/input/mytest.wav --out test/output/x.wav --deep     # deep neural verdict (~1min)
```

## How it works

```
audio.wav → preprocess → STT/LID → ContextLLM → TTS → reply.wav
              (mono/16k/DC/        (faster-whisper   (intent +      (Piper neural
               light-denoise/       auto, 100         memory,        voices per
               norm/trim)            langs)            7 langs)        language)
```

* **Ears** — `polyvoice/preprocess.py` (clean model-ready audio; the denoiser
  can no longer eat speech onsets) and `polyvoice/audio_understand.py`
  (speech, question-like rise, energy, tempo).
* **Brain** — `polyvoice/backends.py:ContextLLM`: greeting/how-are-you/weather/
  name/help/thanks/bye/time/question/statement + silence/acoustic fallbacks,
  per-session memory, replies in en/hi/bn/fr/es/zh/ar (incl. Devanagari and
  Bengali-script keywords). `FasterWhisperSTT` carries a script-consistency
  guard: a `bn` vote with zero Bengali characters and plain English words is
  corrected to English instead of misrouting the reply.
* **Mouth** — `polyvoice/backends.py:PiperTTS`: offline neural voices
  (`voices/`, ~60MB each, downloaded once) with trained-voice fallback.
* **Trained voice** — `polyvoice/universal.py` (any-language adapters, zero-shot
  blends, pitch guard) + `polyvoice/timbre.py` (spectral fingerprint copied
  from real voices, applied as EQ).
* **Verdict** — fast model, deep model (`polyvoice/deep_spoof.py`), or cascade
  combining both (see below).

## Training

| Script | Data | Output |
|---|---|---|
| `train_voice.py` | 64 synthetic utterances, 7 langs | `bank_multi.npz` (incl. Bengali) |
| `train_large.py` | 1,440 mixed: expanded vocab + Tagore/Premchand/Austen novels (`corpora/`, `import_novels.py`) | `bank_large.npz` |
| `train_micro.py` | **3,600 real clips**: GramVaani Hindi, LibriSpeech English, SLR37 Bengali (`data_voice/`, see `fetch_voice_data.py`, `convert_*.py`) | `bank_micro.npz` (params + per-lang EQ residuals + pitch stats) |
| `train_spoof.py` | real humans + 7 Piper voices + hums + shorts + phone/noise/reverb/speed/call-noise/cocktail copies (`--crowd-n`) | `detect.npz` (global) + `detect_hi/en/bn.npz` (per-language) |

Real measured voice stats: Bengali 152.8Hz/wobble 0.19, English 175Hz/0.21,
Hindi 224Hz/0.31. Voice-color distance to real voices after micro-EQ:
Hindi **44% closer**, Bengali **19% closer**. `pytest`: 7 passed.

## Human-vs-AI voice detector (detailed)

**Architecture.** Three per-language detectors (`detect_hi.npz`,
`detect_en.npz`, `detect_bn.npz`) plus one shared fallback (`detect.npz`).
Audio is language-detected automatically (faster-whisper LID) and routed to
its language's model; shaky LID (<0.5 confidence) falls back to shared
weights. Each model is a 19-feature logistic head with duration-aware
(clean/noisy/crowded × short/long) thresholds, multi-window median scoring,
and condition reporting. `--deep` (wav2vec2 detector) and `--cascade`
(fast gate + selective deep confirm) cover borderline clean clips.

**Why per-language?** Measured on training data: English humans are bright
(5.3% high-frequency energy), Hindi humans are band-limited phone audio
(1.0%), Bengali humans are muffled studio (0.6%) — while all AI voices are
bright and full-band. One global boundary must believe "bright = human" and
"bright = AI" simultaneously: a logical contradiction. Splitting by language
resolved it — fresh-AI catch rates went Hindi 7→**15/15**, Bengali 5→**14/15**,
English 12→**15/15**.

**Measured results.**

| Suite | Score | Notes |
|---|---|---|
| 21 scenarios | **20/21** | single miss: one Hindi short |
| Fresh random 49 | **44/49 = 89.8%** | all misses are short clips |
| Unseen 72 (crowds/street/phone) | **60/67 scored = 89.6%** | street/crowd humans hold |
| Fresh AI voices (85 new clips) | **72/85 = 84.7%** | hi/en 15/15, bn 14/15 |
| Tricky 11 (adversarial) | **9/11** | noise/reverb/clip/hum/user-file |
| Per-language held-out | hi **98.3%**, en 89.4%, bn 93.9% | |
| `pytest` | **7 passed** | |

Zero false alarms on real humans — including phone recordings and your own
files — in every suite. Data hygiene via `polyvoice/quality.py` +
`clean_voice_data.py` (82 produced/jingle clips quarantined out of 3,600;
clipped/music/silence checks wired into `fetch_voice_data.py`).

**Known frontier (measured, not guessed):** very short clips (<4s — humans
reach p_ai 0.64, AI sinks to 0.03, best line only 70%; whisper embeddings
agree at 0.89+ cosine), Spanish davefx voice (no measurable traces in 19
statistics or two deep models), 0dB-drowned audio (no voice left to judge).

Test suites (all runnable): `test_all_scenarios.py` (21),
`test_big_eval.py` (67), `test_random_eval.py` (49 fresh),
`test_tricky.py` (11 adversarial), `test_unseen.py` (72 crowds/street),
`gen_fresh_ai.py` builds the 85-clip fresh-AI set (`dataset_ai_fresh/`).

Large files (`voices/` ~436MB, `data_voice/` ~845MB) are intentionally
**not** in git — re-download with `check_voices.py` + `fetch_voice_data.py` /
`convert_librispeech.py` / `convert_hindi.py` (+ SLR37 `bn_in.zip`).

## Honest limits

* Exact words come from Piper's built-in speakers, not a clone of your voice
  (true cloning needs GPU models like XTTS — see `colab_*` helpers).
* Short Bengali clips transcribe transliterated at low confidence (language ID
  itself is reliable); use `--text` for word-exact answers.
* The sine-synth fallback hums prosody; it cannot form words.
* Short audio, drowned (0dB) audio, and davefx-grade voices are documented
  blind spots — the model reports scores, not certainty, there.
