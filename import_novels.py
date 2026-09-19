"""Import public-domain novels (Bengali/Hindi/English) for training vocabulary.

Sources: bn/hi Wikisource (Tagore, Premchand - authors died 1941/1936),
English: Project Gutenberg (Austen, d.1817). Output: corpora/novel_{bn,hi,en}.txt
(one clean sentence per line) + raw dumps under corpora/raw/.
"""
from __future__ import annotations
import os
import re
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) PolyVoiceResearch/1.0"}
OUT = "corpora"

BN_PAGES = [
    "গল্পগুচ্ছ (প্রথম খণ্ড)/কাবুলিওয়ালা",
    "গল্পগুচ্ছ (প্রথম খণ্ড)/ছুটি",
    "গল্পগুচ্ছ (প্রথম খণ্ড)/জয়পরাজয়",
]
HI_PAGES = [
    "मानसरोवर १/ईदगाह",
    "प्रेमचंद रचनावली ६/गोदान-२",
    "प्रेमचंद रचनावली ६/गोदान-३२",
]
EN_URLS = [
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice
]


def fetch(url: str, timeout=60) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _api(base: str, params: dict):
    import json
    import time
    q = urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            time.sleep(1.5)  # be polite to Wikimedia
            return json.loads(fetch(f"{base}/w/api.php?{q}"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:
                time.sleep(5 * (attempt + 1))
                continue
            raise


def _get_wikitext(base: str, titles: list[str]) -> dict[str, str]:
    """Batch-fetch raw wikitext for titles (50 per request)."""
    out: dict[str, str] = {}
    for i in range(0, len(titles), 20):
        chunk = titles[i:i + 50]
        d = _api(base, {"action": "query", "prop": "revisions", "rvprop": "content",
                        "rvslots": "main", "format": "json", "formatversion": 2,
                        "titles": "|".join(chunk)})
        for pg in d.get("query", {}).get("pages", []):
            try:
                out[pg["title"]] = pg["revisions"][0]["slots"]["main"]["content"]
            except (KeyError, IndexError):
                continue
    return out


_PAGE_NS: dict[str, str] = {}


def _page_ns(base: str) -> str:
    if base not in _PAGE_NS:
        d = _api(base, {"action": "query", "meta": "siteinfo",
                        "siprop": "namespaces", "format": "json",
                        "formatversion": 2})
        name = "Page"
        for v in d.get("query", {}).get("namespaces", {}).values():
            if v.get("canonical") == "Page":
                name = v.get("name", "Page")
        _PAGE_NS[base] = name
    return _PAGE_NS[base]


def _local_sys(base: str) -> str:
    if ".bn." in base or "/bn" in base:
        return "beng"
    if ".hi." in base or "/hi" in base:
        return "deva"
    return "arabic"


_DIGITS = {
    "arabic": "0123456789",
    "deva": "०१२३४५६७८९",
    "beng": "০১২৩৪৫৬৭৮৯",
}


def _parse_num(s: str):
    for sys, digits in _DIGITS.items():
        if all(c in digits for c in s):
            return int("".join(str(digits.index(c)) for c in s)), sys
    m = re.search(r"\d+", s)
    return (int(m.group()), "arabic") if m else (None, "arabic")


def _fmt_num(n: int, sys: str) -> str:
    digits = _DIGITS.get(sys, _DIGITS["arabic"])
    return "".join(digits[int(d)] for d in str(n))


def wiki_text(base: str, title: str) -> str:
    got = _get_wikitext(base, [title])
    if title not in got:
        raise ValueError(f"no content for {title}")
    text = got[title]
    # follow <pages index="..pdf" from=.. to=.. /> scan transclusions
    m = re.search(r"<pages\b[^>]*>", text)
    if m:
        tag = m.group()
        idx = re.search(r'index="([^"]+)"', tag)
        fm = re.search(r"from=([^\s/>]+)", tag)
        to = re.search(r"to=([^\s/>]+)", tag)
        if idx and fm and to:
            a, _ = _parse_num(fm.group(1))
            b, _ = _parse_num(to.group(1))
            if a is not None and b is not None and 0 < b - a < 500:
                ns = _page_ns(base)
                loc = _local_sys(base)
                order, want = [], {}
                for n in range(a, b + 1):
                    variants = [f"{ns}:{idx.group(1)}/{_fmt_num(n, s)}"
                                for s in ("arabic", loc)]
                    for t in variants:
                        want[t] = n
                    order.append(n)
                parts = _get_wikitext(base, list(want))
                by_n: dict[int, str] = {}
                for t, n in want.items():
                    if t in parts:
                        by_n.setdefault(n, parts[t])
                text = "\n".join(by_n.get(n, "") for n in order)
    return text


def clean_wiki(t: str) -> str:
    t = re.sub(r"<noinclude>.*?</noinclude>", " ", t, flags=re.S)
    t = re.sub(r"<ref[^>]*>.*?</ref>", " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\{\{[^{}|]*\|([^{}]*)\}\}", r"\1", t)  # keep template display text
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)
    for _ in range(3):  # nested templates/links
        t = re.sub(r"\{\{[^{}]*\}\}", " ", t)
        t = re.sub(r"\[\[[^\[\]|]*\|([^\[\]]*)\]\]", r"\1", t)
        t = re.sub(r"\[\[([^\[\]]*)\]\]", r"\1", t)
    t = re.sub(r"'''?", "", t)
    t = re.sub(r"^==+[^=\n]+==+\s*", "", t, flags=re.M)
    t = re.sub(r"__(?:TOC|NOTOC|NOEDITSECTION)__", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t


def clean_gutenberg(t: str) -> str:
    # strip PG header/footer
    m1 = re.search(r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", t, re.S)
    m2 = re.search(r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", t, re.S)
    if m1:
        t = t[m1.end():]
    if m2:
        t = t[:m2.start()]
    t = re.sub(r"[ \t]+", " ", t)
    return t


SPLIT_RE = re.compile(r"(?<=[।?!.“”\"'’])\s+|\n+")


def sentences(t: str, min_words=3, max_words=30, max_chars=170) -> list[str]:
    out = []
    for s in SPLIT_RE.split(t):
        s = re.sub(r"\s+", " ", s).strip(" \t-–—*\"'“”‘’")
        nw = len(s.split())
        if min_words <= nw <= max_words and len(s) <= max_chars and len(s) >= 12:
            # drop stage directions / list junk / ALL-CAPS headers
            if s.isupper() and len(s) > 20:
                continue
            if re.match(r"^(CHAPTER|Chapter|VOLUME|অধ্যায়|परिच्छेद)\b", s):
                continue
            out.append(s)
    # dedupe, keep order
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def main():
    os.makedirs(f"{OUT}/raw", exist_ok=True)
    jobs = []
    for title in BN_PAGES:
        jobs.append(("bn", title, wiki_text("https://bn.wikisource.org", title), True))
    for title in HI_PAGES:
        jobs.append(("hi", title, wiki_text("https://hi.wikisource.org", title), True))
    for i, url in enumerate(EN_URLS):
        jobs.append(("en", f"gutenberg_{i}", fetch(url), False))

    by_lang: dict[str, list[str]] = {}
    for lang, name, text, is_wiki in jobs:
        raw_p = f"{OUT}/raw/{lang}_{len(by_lang.get(lang, []))}.txt"
        with open(raw_p, "w", encoding="utf-8") as fh:
            fh.write(text)
        t = clean_wiki(text) if is_wiki else clean_gutenberg(text)
        sents = sentences(t)
        by_lang.setdefault(lang, []).extend(sents)
        print(f"{lang} | {name[:40]} | {len(text)//1024}KB raw -> {len(sents)} sentences")

    for lang, sents in by_lang.items():
        p = f"{OUT}/novel_{lang}.txt"
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(sents) + "\n")
        w = sum(len(s.split()) for s in sents)
        print(f"saved {p}: {len(sents)} sentences, {w} words")


if __name__ == "__main__":
    main()
