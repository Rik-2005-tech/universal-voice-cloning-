"""Fresh AI-voice dataset + detector test.

Generates NEW Piper clips (texts never used in training: tail of corpora +
extra sentences) for all 7 voices into dataset_ai_fresh/<lang>/, then scores
every clip with verdict() and writes results to dataset_ai_fresh/RESULTS.txt.
"""
import io
import os
import random
import wave
import numpy as np
from piper import PiperVoice
from polyvoice.spoof import verdict
from polyvoice.backends import PiperTTS

OUT = "dataset_ai_fresh"
PER_VOICE = 15

EXTRA = {
    "en": ["The river flows quietly past the old stone bridge.",
           "She packed her bags before sunrise.", "Every window glowed warm yellow.",
           "The child laughed at the dancing dog.", "Rain drummed on the tin roof all night."],
    "hi": ["नदी पुराने पत्थर के पुल के नीचे बहती है।", "उसने सूरज निकलने से पहले बैग पैक किया।",
           "हर खिड़की में हल्की पीली रोशनी थी।", "बच्चा नाचते कुत्ते को देखकर हंसा।",
           "सारी रात टीन की छत पर बारिश होती रही।"],
    "bn": ["নদী পুরনো পাথরের সেতুর নিচ দিয়ে বয়ে যায়।", "সে ভোরের আগেই ব্যাগ গুছিয়ে নিল।",
           "প্রতিটি জানালায় হালকা হলুদ আলো ছিল।", "বাচ্চাটি নাচতে থাকা কুকুর দেখে হাসল।",
           "সারা রাত টিনের চালে বৃষ্টি পড়ল।"],
    "fr": ["La rivière coule sous le vieux pont de pierre.", "Elle a fait ses valises avant l'aube.",
           "Chaque fenêtre brillait d'une douce lumière.", "L'enfant riait du chien qui dansait.",
           "La pluie a tambouriné sur le toit toute la nuit.", "Le train part à sept heures précises.",
           "J'ai perdu mes clés quelque part.", "Le café du matin sentait très bon.",
           "Nous marchions lentement vers la mer.", "Son sourire illuminait la pièce."],
    "es": ["El río fluye bajo el viejo puente de piedra.", "Hizo las maletas antes del amanecer.",
           "Cada ventana brillaba con luz cálida.", "El niño se reía del perro que bailaba.",
           "La lluvia golpeó el techo toda la noche.", "El tren sale a las siete en punto.",
           "Perdí mis llaves en alguna parte.", "El café olía muy bien.",
           "Caminábamos despacio hacia el mar.", "Su sonrisa iluminaba la habitación."],
    "zh": ["河流静静地流过古老的石桥。", "她在日出前收拾好了行李。",
           "每扇窗户都透出温暖的黄光。", "孩子看着跳舞的小狗笑了。",
           "雨整夜敲打着铁皮屋顶。", "火车七点整准时出发。",
           "我把钥匙丢在什么地方了。", "早晨的咖啡闻起来很香。",
           "我们慢慢地走向大海。", "她的微笑照亮了房间。"],
    "ar": ["يجري النهر بهدوء تحت الجسر الحجري القديم.", "حزمت حقائبها قبل شروق الشمس.",
           "كل نافذة كانت تتوهج بلون أصفر دافئ.", "ضحك الطفل على الكلب الراقص.",
           "قرع المطر السقف طوال الليل.", "يغادر القطار في السابعة تماما.",
           "أضعت مفاتيحي في مكان ما.", "كانت رائحة القهوة طيبة جدا.",
           "مشينا ببطء نحو البحر.", "ابتسامتها أضاءت الغرفة."],
}


def psynth(model, text):
    v = PiperVoice.load(model)
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(v.config.sample_rate)
    v.synthesize_wav(text, w)
    w.close()
    buf.seek(0)
    r = wave.open(buf, "rb")
    x = (np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16).astype(np.float32)
         / 32768.0)
    return x, r.getframerate()


def main():
    import soundfile as sf
    rng = random.Random(99)  # different seed from training (0) -> unseen texts
    for lang, model in sorted(PiperTTS.VOICE_MAP.items()):
        d = os.path.join(OUT, lang)
        os.makedirs(d, exist_ok=True)
        pool = []
        corp = f"corpora/novel_{lang}.txt"
        if os.path.exists(corp):
            with open(corp, encoding="utf-8") as fh:
                lines = [l.strip() for l in fh if 30 < len(l.strip()) < 150]
            rng.shuffle(lines)
            pool += lines[-40:]  # tail: never used in training head-samples
        pool += EXTRA.get(lang) or EXTRA["en"]
        rng.shuffle(pool)
        n = 0
        for t in pool[:PER_VOICE]:
            try:
                x, sr = psynth(model, t[:160])
                sf.write(os.path.join(d, f"{lang}_fresh_{n:03d}.wav"), x, sr)
                with open(os.path.join(d, f"{lang}_fresh_{n:03d}.txt"), "w",
                          encoding="utf-8") as fh:
                    fh.write(t[:160])
                n += 1
            except Exception as e:
                print(f"skip {lang}: {str(e)[:60]}", flush=True)
        print(f"generated {lang}: {n}", flush=True)
    # ---- test ----
    ok, tot, per = 0, 0, {}
    lines = []
    for lang in sorted(PiperTTS.VOICE_MAP):
        d = os.path.join(OUT, lang)
        g = b = 0
        for f in sorted(os.listdir(d)):
            if not f.endswith(".wav"):
                continue
            x, sr = sf.read(os.path.join(d, f), always_2d=False)
            v = verdict(np.asarray(x), sr)
            good = v["label"] == "AI"
            g += good
            b += 1
            ok += good
            tot += 1
        per[lang] = (g, b)
        lines.append(f"{lang}: {g}/{b} AI caught")
        print(f"{lang}: {g}/{b} AI caught", flush=True)
    lines.append(f"TOTAL {ok}/{tot} = {ok/tot:.3f}")
    print(f"TOTAL {ok}/{tot} = {ok/tot:.3f}", flush=True)
    with open(os.path.join(OUT, "RESULTS.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
