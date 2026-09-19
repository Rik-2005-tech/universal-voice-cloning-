"""Train universal any-language voice: python3 train_voice.py --bank-out bank.npz [--data data/]

Layouts for --data (ANY language codes, ANY count):
  data/<lang>/*.wav (+ optional same-name .txt)   e.g. data/sw/*.wav, data/yo/*.wav
  data/*.wav with <lang>_*.wav prefix             e.g. data/zh_001.wav
Without --data: synthetic multilingual demo (en/fr/es/hi/zh/ar) offline.

Full GPU recipe (A100/H100):
 1) data: 50-500h multilingual 24kHz (LibriTTS-R, CommonVoice 17, MLS + your voice, consent)
 2) base: Coqui XTTS-v2 or MMS-VITS; per-lang LoRA rank-8 adapters + shared backbone
 3) frontend: phonemizer/IPA + universal.analyze_text (script/tonal/rhythm)
 4) adversarial: ASVspoof5 detector alongside; humanize layer always on
 5) eval: MOS>=4.5, WER<=human, spoof gap <0.08, ABX ~50%. Watermark AudioSeal.
This file: CPU smoke-train of the same API (global + per-lang adapters).
"""
from __future__ import annotations
import argparse
import numpy as np


def synth_human_ref(text: str, sr=16000, seed=0) -> np.ndarray:
    """Human-like reference: formant resonances + shimmer + aspiration +
    consonant bursts. Matches universal synth timbre (16 harmonics, frication
    texture) so smoke-training optimizes meaningfully."""
    rng = np.random.default_rng(seed)
    dur = max(0.5, len(text) / 13.0)
    n = int(sr * dur)
    t = np.arange(n) / sr
    f0 = 172 + 8 * np.sin(2 * np.pi * 0.9 * t) + rng.standard_normal(n) * 0.4
    phase = 2 * np.pi * np.cumsum(f0) / sr + 2.5 * np.sin(2 * np.pi * 5.2 * t)
    wav = np.zeros(n, dtype=np.float64)
    for k in range(1, 17):
        fk = 172 * k
        w = (1.0 + 0.9 * np.exp(-((fk - 500) / 450) ** 2)
             + 0.7 * np.exp(-((fk - 1500) / 650) ** 2))
        amp = (0.32 / (k ** 0.9)) * w / 1.8
        if k > 8:
            amp *= 0.5
        wav = wav + amp * np.sin(k * phase + 0.3 * k)
    wav = wav * (1.0 + 0.02 * np.sin(2 * np.pi * 7 * t))
    asp = rng.standard_normal(n)
    asp = (asp + np.concatenate([[0], asp[:-1]])) * 0.5
    wav = wav + asp * 0.02
    # consonant-like frication bursts per word (mirrors synth, subtle)
    nw = max(1, min(len(text.split()), 30))
    grid = np.linspace(0.05, 0.95, nw) * n + rng.uniform(-0.02, 0.02, nw) * n
    blen = max(8, int(sr * 0.02))
    benv = np.hanning(blen)
    for c in np.clip(grid.astype(int), 0, max(0, n - blen)):
        wav[c:c + blen] += rng.standard_normal(blen) * benv * 0.03
    return wav.astype(np.float32)


DEMO_BANK_TEXTS = {
    # Expanded any-situation bank: greeting / small-talk / question / help /
    # thanks / bye / time / weather / happy / empathetic / long-form / short.
    "en": [
        "Hello, how can I help you today?",
        "The weather is wonderful this morning.",
        "Good morning, welcome to the voice cloning demo.",
        "How are you doing this beautiful afternoon?",
        "What is your name and how can you help me?",
        "Thank you very much, goodbye!",
        "Could you please speak slowly and clearly?",
        "I am so happy to hear the great news today!",
        "I am sorry you feel sad, take your time.",
        "What time is it right now, please?",
        "The quick brown fox jumps over the lazy dog.",
        "Streaming synthesis must start in under half a second.",
    ],
    "fr": [
        "Bonjour, comment vas-tu?",
        "Je vais très bien, merci beaucoup.",
        "Quel temps fait-il aujourd'hui s'il vous plaît?",
        "Comment puis-je vous aider ce matin?",
        "Merci beaucoup, au revoir!",
        "Parlez lentement et clairement s'il vous plaît.",
        "Je suis très heureux d'apprendre cette bonne nouvelle!",
        "Je suis désolé, prenez votre temps.",
    ],
    "es": [
        "Hola, ¿cómo puedo ayudarte?",
        "El clima está maravilloso hoy.",
        "¿Cómo estás esta hermosa tarde?",
        "¿Cuál es tu nombre y cómo puedes ayudarme?",
        "Muchas gracias, ¡adiós!",
        "Habla despacio y con claridad, por favor.",
        "¡Estoy muy feliz de escuchar esta gran noticia!",
        "Lo siento mucho, tómate tu tiempo.",
    ],
    "hi": [
        "Namaste, main aapki kaise madad kar sakta hun?",
        "Mausam aaj bahut achha hai.",
        "Aap kaise hain aaj dopahar?",
        "Aapka naam kya hai?",
        "Bahut bahut dhanyavad, alvida!",
        "Kripya dheere aur spasht bolein.",
        "Yeh khabar sunkar main bahut khush hun!",
        "Mujhe khed hai, apna samay lein.",
        "नमस्ते! आप कैसे हैं?",
        "आज मौसम बहुत अच्छा है।",
        "आपका नाम क्या है?",
        "बहुत-बहुत धन्यवाद, अलविदा!",
    ],
    "zh": [
        "你好，你今天怎么样？",
        "今天天气非常好。",
        "请问你叫什么名字？",
        "请慢慢说，清楚一点。",
        "非常感谢，再见！",
        "听到这个好消息我非常高兴！",
        "很抱歉，请慢慢来。",
        "现在几点钟了？",
    ],
    "ar": [
        "مرحبا، كيف حالك اليوم؟",
        "الطقس جميل جدا هذا الصباح.",
        "ما اسمك وكيف يمكنك مساعدتي؟",
        "تكلم ببطء وبوضوح من فضلك.",
        "شكرا جزيلا، وداعا!",
        "أنا سعيد جدا لسماع هذه الأخبار!",
        "أنا آسف، خذ وقتك.",
        "كم الساعة الآن من فضلك؟",
    ],
    "bn": [
        "নমস্কার! আপনি কেমন আছেন?",
        "আজ আবহাওয়া খুব ভালো।",
        "আপনার নাম কী?",
        "অনেক অনেক ধন্যবাদ, বিদায়!",
        "দয়া করে ধীরে এবং স্পষ্ট করে বলুন।",
        "এই খবর শুনে আমি খুব খুশি!",
        "দুঃখিত, আপনার সময় নিন।",
        "এখন কয়টা বাজে?",
    ],
}


def main():
    ap = argparse.ArgumentParser(description="PolyVoice universal any-language trainer")
    ap.add_argument("--out", default="voice.npz", help="legacy single-voice ckpt out")
    ap.add_argument("--bank-out", default="bank.npz", help="multilingual adapter bank out")
    ap.add_argument("--data", default=None, help="training data dir (any <lang>/*.wav layout)")
    ap.add_argument("--steps", type=int, default=25, help="steps for legacy single-voice mode")
    ap.add_argument("--steps-global", type=int, default=15)
    ap.add_argument("--steps-per-lang", type=int, default=8)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--single", action="store_true", help="also save legacy single voice.npz")
    args = ap.parse_args()

    from polyvoice.train_superhuman import train_voice, train_multilingual, save_bank
    from polyvoice.universal import load_training_bank

    if args.data:
        bank = load_training_bank(args.data, sr=args.sr)
        if not bank:
            raise SystemExit(f"no wavs found in {args.data} (expect <lang>/*.wav or <lang>_*.wav)")
        print(f"loaded langs: {sorted(bank)} counts={[len(v[0]) for v in bank.values()]}")
    else:
        bank = {}
        for i, (lang, texts) in enumerate(DEMO_BANK_TEXTS.items()):
            refs = [synth_human_ref(t, sr=args.sr, seed=100 + i * 10 + j) for j, t in enumerate(texts)]
            bank[lang] = (texts, refs)
        print(f"demo bank langs: {sorted(bank)} (use --data for real speech)")

    per, glob, hists = train_multilingual(
        bank, steps_global=args.steps_global, steps_per_lang=args.steps_per_lang,
        sr=args.sr)
    for lang, h in hists.items():
        print(f"[{lang}] loss {h[0]:.4f} -> {h[-1]:.4f}")
    save_bank(args.bank_out, per)
    print(f"saved bank {args.bank_out}: langs={sorted(per)} global={glob}")

    if args.single or not args.data:
        # legacy compat: flatten bank -> single voice
        all_texts, all_refs = [], []
        for texts, refs in bank.values():
            all_texts.extend(texts)
            all_refs.extend(refs)
        p, hist = train_voice(all_texts, all_refs, steps=args.steps, sr=args.sr)
        print(f"single loss {hist[0]:.4f} -> {hist[-1]:.4f}")
        np.savez(args.out, f0=p.f0, vib_depth=p.vib_depth, vib_rate=p.vib_rate,
                 brightness=p.brightness, breath_db=p.breath_db, rate=p.rate)
        print(f"saved {args.out}: {p}")

    # eval: seen lang + zero-shot unseen lang (e.g. 'sw' not in demo bank)
    from polyvoice.universal import synth_universal, adapt_to_lang, analyze_text
    from polyvoice.humanize import humanize
    from polyvoice.eval_indist import evaluate_pair
    seen = sorted(bank)[0]
    p_seen = adapt_to_lang(seen, per, glob)
    tx = bank[seen][0][0]
    ref = bank[seen][1][0]
    hyp = humanize(synth_universal(tx, seen, p_seen, sr=args.sr), args.sr, tx)
    print(f"eval seen [{seen}] {analyze_text(tx, seen)}:", evaluate_pair(ref[:len(hyp)], hyp))
    p_unseen = adapt_to_lang("sw", per, glob)  # Swahili likely unseen -> zero-shot blend
    hyp2 = humanize(synth_universal("Habari, habari za asubuhi?", "sw", p_unseen, sr=args.sr),
                    args.sr, "Habari?", f0=188.0)
    print(f"zero-shot [sw] params={p_unseen} len={len(hyp2)}")

if __name__ == "__main__":
    main()
