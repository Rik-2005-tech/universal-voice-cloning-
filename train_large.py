"""Large-data long training until converged (minibatch, early stopping)."""
from __future__ import annotations
import numpy as np


def build_large_bank(sr=16000, per_lang=100, seed=0):
    from train_voice import synth_human_ref
    rng = np.random.default_rng(seed)

    en_tpl = [
        "Hello, how can I help you today?", "Good morning, welcome aboard.",
        "How are you doing this beautiful {w}?", "What is your name, please?",
        "Thank you very much, goodbye!", "Could you speak slowly and clearly, please?",
        "I am so happy to hear the {w} news!", "I am sorry you feel {w}, take your time.",
        "What time is it right now?", "Where is the nearest {w} station, please?",
        "Please help me with my {w} reservation.", "The {w} weather today is {w2}.",
        "I missed my {w}, what should I do now?", "Tell me more about the {w} plan.",
        "Can you repeat that {w} again?", "This is an urgent {w} situation.",
    ]
    fr_tpl = [
        "Bonjour, comment puis-je vous aider?", "Quel temps fait-il ce {w}?",
        "Comment allez-vous cet {w}?", "Merci beaucoup, au revoir!",
        "Parlez lentement et clairement, s'il vous plait.", "Ou est la gare {w}, s'il vous plait?",
        "J'ai perdu mes {w}, aidez-moi!", "Je suis tres heureux de cette {w} nouvelle!",
        "Je suis desole, prenez votre {w}.", "Quelle heure est-il {w}?",
    ]
    es_tpl = [
        "Hola, como puedo ayudarte?", "Como estas esta {w} tarde?",
        "Muchas gracias, adios!", "Habla despacio y con claridad, por favor.",
        "Donde esta la estacion {w}?", "He perdido mi {w}, ayudame!",
        "Estoy muy feliz por la noticia {w}!", "Lo siento, tomate tu {w}.",
        "Que hora es {w}?", "Necesito ayuda urgente con mi {w}.",
    ]
    hi_tpl = [
        "Namaste, main aapki kaise madad kar sakta hun?", "Aap kaise hain aaj {w}?",
        "Bahut dhanyavad, alvida!", "Kripya dheere aur spasht bolein {w}.",
        "Meri train chhoot gayi, ab kya karun?", "Yeh {w} sunkar main bahut khush hun!",
        "Mujhe khed hai, apna {w} lein.", "Sahayata ke liye {w} dabayein.",
        "Mausam aaj bahut {w} hai.", "Kya samay hua hai {w}?",
    ]
    zh_tpl = [
        "你好，你今天怎么样？", "今天天气非常好。",
        "请问你叫什么名字？", "请慢慢说，清楚一点。",
        "非常感谢，再见！", "听到这个好消息我非常高兴！",
        "很抱歉，请慢慢来。", "现在几点钟了？",
        "我的钱包丢了，现在该怎么办？", "请帮我查一下车站怎么走。",
    ]
    ar_tpl = [
        "مرحبا، كيف حالك اليوم؟", "الطقس جميل جدا هذا الصباح.",
        "ما اسمك وكيف يمكنك مساعدتي؟", "تكلم ببطء وبوضوح من فضلك.",
        "شكرا جزيلا، وداعا!", "أنا سعيد جدا لسماع هذه الأخبار!",
        "أنا آسف، خذ وقتك.", "كم الساعة الآن من فضلك؟",
        "فقدت جواز سفري ماذا أفعل؟", "أين أقرب محطة من فضلك؟",
    ]
    fill_w = ["morning", "evening", "beautiful", "urgent", "quick", "bright",
              "early", "late", "daily", "local", "central", "grand"]
    fill_w2 = ["wonderful", "pleasant", "cloudy", "sunny", "calm", "busy"]
    tpl_map = {"en": en_tpl, "fr": fr_tpl, "es": es_tpl,
               "hi": hi_tpl, "zh": zh_tpl, "ar": ar_tpl}

    bank = {}
    for li, (lang, tpls) in enumerate(tpl_map.items()):
        texts, refs = [], []
        for i in range(per_lang):
            t = tpls[i % len(tpls)]
            if "{w}" in t:
                t = t.replace("{w}", fill_w[rng.integers(0, len(fill_w))])
            if "{w2}" in t:
                t = t.replace("{w2}", fill_w2[rng.integers(0, len(fill_w2))])
            if i >= len(tpls):  # make later copies distinct situations
                t = f"{t} (case {i})"
            texts.append(t)
            refs.append(synth_human_ref(t, sr=sr, seed=1000 + li * 500 + i))
        bank[lang] = (texts, refs)
    return bank


def train_minibatch(texts, refs, langs, steps=200, batch=10, lr=0.08,
                    sr=16000, seed=0, init=None, val_n=48, patience=6,
                    tag="global"):
    from polyvoice.train_superhuman import VoiceParams, loss_vs_ref, mel_like
    rng = np.random.default_rng(seed)
    refs_s = [np.asarray(r, dtype=np.float32)[: int(sr * 1.2)] for r in refs]
    ref_mels = [mel_like(r, n_fft=256, n_bands=24) for r in refs_s]
    n = len(texts)
    vidx = rng.choice(n, size=min(val_n, n), replace=False)
    v = init.vector().copy() if init is not None else VoiceParams().vector()
    best, best_v = 1e9, v.copy()
    hist, bad = [], 0
    for s in range(steps):
        bidx = rng.choice(n, size=min(batch, n), replace=False)
        bt = [texts[i] for i in bidx]
        br = [refs_s[i] for i in bidx]
        bm = [ref_mels[i] for i in bidx]
        bl = [langs[i] for i in bidx]
        cur = float(np.mean([loss_vs_ref(t, VoiceParams.from_vector(v), r, sr, rm, l)
                             for t, r, rm, l in zip(bt, br, bm, bl)]))
        i = int(rng.integers(0, len(v)))
        eps = 0.03
        dv = v.copy()
        dv[i] += eps
        lp = float(np.mean([loss_vs_ref(t, VoiceParams.from_vector(dv), r, sr, rm, l)
                            for t, r, rm, l in zip(bt, br, bm, bl)]))
        g = (lp - cur) / eps
        v[i] = v[i] - lr * float(np.clip(g, -2, 2))
        v[3] = float(np.clip(v[3], 0.05, 0.95))
        v[5] = float(np.clip(v[5], 0.8, 1.25))
        v[0] = float(np.clip(v[0], 0.6, 1.3))
        if (s + 1) % 10 == 0 or s == 0:
            val = float(np.mean([loss_vs_ref(texts[i], VoiceParams.from_vector(v),
                                             refs_s[i], sr, ref_mels[i], langs[i])
                                 for i in vidx]))
            hist.append(val)
            if val < best - 1e-4:
                best, best_v = val, v.copy()
                bad = 0
            else:
                bad += 1
            print(f"[{tag}] step {s+1}/{steps} batch_loss={cur:.4f} val={val:.4f} best={best:.4f}",
                  flush=True)
            if bad >= patience:
                print(f"[{tag}] converged (early stop at step {s+1})", flush=True)
                break
    return VoiceParams.from_vector(best_v), hist


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=100)
    ap.add_argument("--steps-global", type=int, default=200)
    ap.add_argument("--steps-per-lang", type=int, default=30)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--bank-out", default="bank_large.npz")
    ap.add_argument("--sr", type=int, default=16000)
    a = ap.parse_args()

    from polyvoice.train_superhuman import save_bank

    print(f"building large bank: {a.per_lang}/lang ...", flush=True)
    bank = build_large_bank(sr=a.sr, per_lang=a.per_lang)
    total = sum(len(v[0]) for v in bank.values())
    print(f"bank langs={sorted(bank)} total={total}", flush=True)

    all_texts, all_refs, all_langs = [], [], []
    for lang, (texts, refs) in bank.items():
        all_texts.extend(texts)
        all_refs.extend(refs)
        all_langs.extend([lang] * len(texts))

    glob, gh = train_minibatch(all_texts, all_refs, all_langs,
                               steps=a.steps_global, batch=a.batch,
                               sr=a.sr, seed=0, tag="global")
    print(f"global val {gh[0]:.4f} -> {gh[-1]:.4f}", flush=True)

    per, hists = {}, {}
    for li, (lang, (texts, refs)) in enumerate(sorted(bank.items())):
        p, h = train_minibatch(texts, refs, [lang] * len(texts),
                               steps=a.steps_per_lang, batch=min(a.batch, 8),
                               sr=a.sr, seed=100 + li, init=glob,
                               tag=f"lang:{lang}")
        import numpy as np
        v = 0.5 * glob.vector() + 0.5 * p.vector()
        from polyvoice.train_superhuman import VoiceParams
        per[lang] = VoiceParams.from_vector(v)
        hists[lang] = h
        print(f"[{lang}] {h[0]:.4f} -> {h[-1]:.4f}", flush=True)
    per.setdefault("und", glob)
    save_bank(a.bank_out, per)
    print(f"saved {a.bank_out}: langs={sorted(per)} global={glob}", flush=True)


if __name__ == "__main__":
    main()
