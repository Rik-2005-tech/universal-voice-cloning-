"""PolyVoice — streaming audio-in / audio-out multilingual voice model."""
from .pipeline import PolyVoicePipeline, PipelineConfig
from .audio_io import normalize_loudness, denoise_spectral_gate, vad_energy
from .backends import DummySTT, DummyLLM, DummyTTS, SuperhumanTTS, TranscriptSTT, AcousticSTT, ContextLLM, PiperTTS, HybridLLM, TransformersLLM
from .audio_understand import analyze_audio
from .preprocess import preprocess_audio
from .timbre import apply_eq, f0_stats, learn_residual, load_micro_bank
from .deep_spoof import verdict_deep
from .spoof import verdict_auto
from .universal import analyze_text, lang_embedding, synth_universal, adapt_to_lang, load_training_bank

__all__ = [
    "PolyVoicePipeline",
    "PipelineConfig",
    "normalize_loudness",
    "denoise_spectral_gate",
    "vad_energy",
    "DummySTT",
    "DummyLLM",
    "DummyTTS",
    "SuperhumanTTS",
    "TranscriptSTT",
    "AcousticSTT",
    "ContextLLM",
    "PiperTTS",
    "HybridLLM",
    "TransformersLLM",
    "analyze_audio",
    "preprocess_audio",
    "apply_eq",
    "f0_stats",
    "learn_residual",
    "load_micro_bank",
    "verdict_deep",
    "verdict_auto",
    "analyze_text",
    "lang_embedding",
    "synth_universal",
    "adapt_to_lang",
    "load_training_bank",
]
