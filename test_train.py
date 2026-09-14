"""Training + indistinguishability tests (CPU, seconds)."""
import numpy as np


def test_train_loss_drops():
    from polyvoice.train_superhuman import train_voice
    from train_voice import synth_human_ref
    texts = ["Hello test one", "Bonjour test deux"]
    refs = [synth_human_ref(t, seed=i) for i, t in enumerate(texts)]
    _, hist = train_voice(texts, refs, steps=12)
    assert hist[-1] < hist[0], f"loss should drop: {hist[0]:.4f} -> {hist[-1]:.4f}"


def test_humanize_and_eval():
    from polyvoice.train_superhuman import VoiceParams, synth_from_params
    from polyvoice.humanize import humanize
    from polyvoice.eval_indist import evaluate_pair
    from train_voice import synth_human_ref
    ref = synth_human_ref("Hello world test", seed=0)
    hyp = humanize(synth_from_params("Hello world test", VoiceParams(), sr=16000), 16000, "Hello world test")
    e = evaluate_pair(ref[:len(hyp)], hyp)
    assert e["mel_L2"] < 2.0 and e["f0_rmse"] < 120.0, e


def test_superhuman_tts_loads_ckpt(tmp_path=None):
    from polyvoice.backends import SuperhumanTTS
    t = SuperhumanTTS(ckpt="/tmp/voice.npz")
    wav, sr = t.synth("Bonjour, comment vas-tu?", "fr")
    assert sr == 24000 and len(wav) > 4000
