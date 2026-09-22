# PolyVoice — offline multilingual talking assistant

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

Voice check on any file:

```bash
python3 app_cli.py --in test/input/mytest.wav --out test/output/x.wav --detect          # fast (<1s)
python3 app_cli.py --in test/input/mytest.wav --out test/output/x.wav --cascade        # fast + auto deep confirm
python3 app_cli.py --in test/input/mytest.wav --out test/output/x.wav --deep           # deep neural verdict (~1min)
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
  Bengali-script keywords).
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
| `train_spoof.py` | real humans + 7 Piper voices + hums + shorts + phone/noise/reverb/speed/call-noise/cocktail copies (`--crowd-n`) | `detect.npz` (weights + duration/condition thresholds) |

Real measured voice stats: Bengali 152.8Hz/wobble 0.19, English 175Hz/0.21,
Hindi 224Hz/0.31. Voice-color distance to real voices after micro-EQ:
Hindi **44% closer**, Bengali **19% closer**. `pytest`: 7 passed.

## Human-vs-AI voice detector

The fast detector (`polyvoice/spoof.py`, weights `detect.npz`) uses 19 acoustic
features (incl. vocoder-artifact bands: crest, high-band flatness, 6kHz+
energy, contrast, group-delay roughness, hiss burstiness) with
condition-aware thresholds (clean/noisy/crowded × short/long, multi-window
median scoring). Data hygiene via `polyvoice/quality.py` +
`clean_voice_data.py` (82 produced/jingle clips quarantined out of 3,600;
clipped/music/silence checks wired into `fetch_voice_data.py`).

Measured: **19/21 scenarios, 43/49 fresh-random**, zero false alarms on
real humans incl. phone recordings. Known frontier: very short clips
(<4s, human and AI provably overlap — best line only 70%) and the Spanish
davefx voice (fools every detector we own, deep ones included).

Test suites (all runnable): `test_all_scenarios.py` (21),
`test_big_eval.py` (67), `test_random_eval.py` (49 fresh),
`test_tricky.py` (11 adversarial: noise, reverb, clipping, hums).

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
