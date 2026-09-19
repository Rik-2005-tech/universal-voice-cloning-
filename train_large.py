"""Large-data long training until converged (minibatch, early stopping)."""
from __future__ import annotations
import numpy as np


def build_large_bank(sr=16000, per_lang=100, per_lang_main=250, seed=0,
                     main_langs=("en", "hi", "bn"), novel_dir="corpora",
                     novel_per_lang=150, novel_langs=("en", "hi", "bn")):
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
        # --- extra vocabulary: travel, food, health, shopping, directions ---
        "I would like to book a {w} ticket to the city.",
        "How much does this {w} cost?",
        "Where can I find a good {w} restaurant nearby?",
        "I need a doctor, it is a {w} emergency.",
        "Could you show me the way to the {w} market?",
        "My luggage is lost at the {w} airport.",
        "What is the phone number of the {w} hotel?",
        "I feel {w} today, thank you for asking.",
        "Please call a taxi for the {w} morning.",
        "The train leaves at {w} o'clock sharp.",
        "I love the {w} festivals in this season.",
        "Can you help me fill this {w} form?",
        "Yesterday the {w} meeting was very long.",
        "Tomorrow will be a {w} and sunny day.",
        "My family lives in a small {w} village.",
        "This book about {w} history is fascinating.",
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
        "नमस्ते! आप कैसे हैं?", "आज मौसम बहुत {w} है।",
        "आपका नाम क्या है?", "बहुत-बहुत धन्यवाद, अलविदा!",
        # --- extra vocabulary: travel, food, health, shopping, directions ---
        "Mujhe sheher ke liye {w} ticket book karna hai.",
        "Yeh {w} kitne ka hai?",
        "Aas-paas achha {w} restaurant kahan milega?",
        "Mujhe doctor chahiye, {w} emergency hai.",
        "Kripya mujhe {w} bazaar ka rasta dikhayein.",
        "Hawai adde par mera {w} saman kho gaya.",
        "{w} hotel ka phone number kya hai?",
        "Aaj main {w} mehsoos kar raha hun.",
        "Kripya {w} subah ke liye taxi bulayein.",
        "Train bilkul {w} baje chhootti hai.",
        "Mujhe is mausam ke {w} tyohar pasand hain.",
        "Kya aap mera {w} form bharne mein madad karenge?",
        "Kal ki {w} meeting bahut lambi thi.",
        "Mera parivar ek chhote {w} gaon mein rehta hai.",
        "Doctor ne mujhe {w} dawai di hai.",
        "Bazaar se {w} sabzi kharidkar laayein.",
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
    bn_tpl = [
        "নমস্কার! আপনি কেমন আছেন?", "আজ আবহাওয়া খুব ভালো।",
        "আপনার নাম কী?", "অনেক অনেক ধন্যবাদ, বিদায়!",
        "দয়া করে ধীরে এবং স্পষ্ট করে বলুন।", "এই খবর শুনে আমি খুব খুশি!",
        "দুঃখিত, আপনার সময় নিন।", "এখন কয়টা বাজে?",
        "আমি কীভাবে সাহায্য করতে পারি?", "আপনার শহরের নাম বলুন।",
        # --- extra vocabulary: travel, food, health, shopping, directions ---
        "আমি শহরে যাওয়ার জন্য {w} টিকিট বুক করতে চাই।",
        "এই {w} টার দাম কত?",
        "কাছাকাছি ভালো {w} রেস্তোরাঁ কোথায় পাব?",
        "আমার ডাক্তার দরকার, এটি {w} জরুরি অবস্থা।",
        "দয়া করে আমাকে {w} বাজারের রাস্তা দেখান।",
        "বিমানবন্দরে আমার {w} লাগেজ হারিয়ে গেছে।",
        "{w} হোটেলের ফোন নম্বর কত?",
        "আজ আমার {w} লাগছে, জিজ্ঞেস করার জন্য ধন্যবাদ।",
        "দয়া করে {w} সকালের জন্য ট্যাক্সি ডাকুন।",
        "ট্রেন ঠিক {w} টায় ছাড়বে।",
        "এই মৌসুমের {w} উৎসব আমার ভালো লাগে।",
        "আপনি কি আমার {w} ফর্ম পূরণে সাহায্য করবেন?",
        "গতকালের {w} মিটিং অনেক লম্বা ছিল।",
        "আমার পরিবার ছোট {w} গ্রামে থাকে।",
        "ডাক্তার আমাকে {w} ওষুধ দিয়েছেন।",
        "বাজার থেকে {w} সবজি কিনে আনুন।",
    ]
    # per-language vocab pools (topics: time, place, food, mood, travel)
    FILL = {
        "en": ["morning", "evening", "beautiful", "urgent", "quick", "bright",
               "early", "late", "daily", "local", "central", "grand",
               "railway", "happy", "sad", "busy", "quiet", "crowded",
               "ticket", "lunch", "dinner", "doctor", "hotel", "market",
               "airport", "school", "hospital", "garden", "river", "mountain",
               "monday", "friday", "summer", "winter", "fresh", "healthy"],
        "hi": ["subah", "shaam", "sundar", "turant", "jaldi", "ujjwal",
               "railway", "khush", "udaas", "vyast", "shaant", "bheed",
               "ticket", "khana", "doctor", "hotel", "bazaar", "school",
               "aspataal", "bagicha", "nadi", "pahaad", "somvar", "shukravar",
               "garmi", "sardi", "taaza", "swasth", "parivar", "dost"],
        "bn": ["সকাল", "সন্ধ্যা", "সুন্দর", "জরুরি", "তাড়াতাড়ি", "উজ্জ্বল",
               "রেল", "খুশি", "দুঃখিত", "ব্যস্ত", "শান্ত", "ভিড়",
               "টিকিট", "খাবার", "ডাক্তার", "হোটেল", "বাজার", "স্কুল",
               "হাসপাতাল", "বাগান", "নদী", "পাহাড়", "সোমবার", "শুক্রবার",
               "গ্রীষ্ম", "শীত", "তাজা", "সুস্থ", "পরিবার", "বন্ধু"],
    }
    FILL2 = {
        "en": ["wonderful", "pleasant", "cloudy", "sunny", "calm", "busy"],
        "hi": ["suhana", "khushnuma", "badli", "dhoop", "shaant", "vyast"],
        "bn": ["চমৎকার", "মনোরম", "মেঘলা", "রৌদ্রোজ্জ্বল", "শান্ত", "ব্যস্ত"],
    }
    fill_w = ["morning", "evening", "beautiful", "urgent", "quick", "bright",
              "early", "late", "daily", "local", "central", "grand"]
    fill_w2 = ["wonderful", "pleasant", "cloudy", "sunny", "calm", "busy"]
    tpl_map = {"en": en_tpl, "fr": fr_tpl, "es": es_tpl,
               "hi": hi_tpl, "zh": zh_tpl, "ar": ar_tpl, "bn": bn_tpl}

    bank = {}
    for li, (lang, tpls) in enumerate(tpl_map.items()):
        n_lang = per_lang_main if lang in main_langs else per_lang
        fw = FILL.get(lang, fill_w)
        fw2 = FILL2.get(lang, fill_w2)
        texts, refs = [], []
        for i in range(n_lang):
            t = tpls[i % len(tpls)]
            if "{w}" in t:
                t = t.replace("{w}", fw[rng.integers(0, len(fw))])
            if "{w2}" in t:
                t = t.replace("{w2}", fw2[rng.integers(0, len(fw2))])
            if i >= len(tpls):  # make later copies distinct situations
                t = f"{t} (case {i})"
            texts.append(t)
            refs.append(synth_human_ref(t, sr=sr, seed=1000 + li * 500 + i))
        bank[lang] = (texts, refs)
    # --- real novel sentences (Tagore/Premchand/Austen): true vocabulary ---
    import os as _os
    for lang in novel_langs:
        p = _os.path.join(novel_dir or "", f"novel_{lang}.txt")
        if not (novel_dir and _os.path.exists(p)):
            continue
        with open(p, encoding="utf-8") as fh:
            pool = [l.strip() for l in fh if l.strip()]
        if not pool:
            continue
        k = min(novel_per_lang, len(pool))
        pick = rng.choice(len(pool), size=k, replace=False)
        texts, refs = bank.get(lang, ([], []))
        base_seed = 9000 + {"en": 0, "hi": 1000, "bn": 2000}.get(lang, 3000)
        for j, idx in enumerate(sorted(pick)):
            t = pool[int(idx)]
            texts.append(t)
            refs.append(synth_human_ref(t, sr=sr, seed=base_seed + j))
        bank[lang] = (texts, refs)
        print(f"novel {lang}: +{k} sentences from {p}", flush=True)
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
    ap.add_argument("--per-lang", type=int, default=60)
    ap.add_argument("--per-lang-main", type=int, default=250,
                    help="utterances for main langs (en/hi/bn)")
    ap.add_argument("--novel-dir", default="corpora",
                    help="dir with novel_{en,hi,bn}.txt (empty to skip)")
    ap.add_argument("--novel-per-lang", type=int, default=150,
                    help="novel sentences mixed in per novel lang")
    ap.add_argument("--data-dir", default=None,
                    help="real voice dataset dir (subfolders per lang, .wav + .txt) e.g. data_voice")
    ap.add_argument("--data-per-lang", type=int, default=1200,
                    help="max real clips per lang to mix in")
    ap.add_argument("--steps-global", type=int, default=200)
    ap.add_argument("--steps-per-lang", type=int, default=30)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--bank-out", default="bank_large.npz")
    ap.add_argument("--sr", type=int, default=16000)
    a = ap.parse_args()

    from polyvoice.train_superhuman import save_bank

    print(f"building large bank: {a.per_lang}/lang ({a.per_lang_main} for en/hi/bn) ...", flush=True)
    bank = build_large_bank(sr=a.sr, per_lang=a.per_lang, per_lang_main=a.per_lang_main,
                            novel_dir=a.novel_dir, novel_per_lang=a.novel_per_lang)
    if a.data_dir:
        import numpy as _np
        from polyvoice.universal import load_training_bank
        real = load_training_bank(a.data_dir, sr=a.sr)
        _rrng = _np.random.default_rng(7)
        for lang, (texts, refs) in real.items():
            k = min(a.data_per_lang, len(texts))
            idx = _rrng.choice(len(texts), size=k, replace=False)
            rt = [texts[i] for i in sorted(idx)]
            rr = [refs[i] for i in sorted(idx)]
            t0, r0 = bank.get(lang, ([], []))
            bank[lang] = (t0 + rt, r0 + rr)
            print(f"real {lang}: +{k} clips from {a.data_dir}", flush=True)
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
