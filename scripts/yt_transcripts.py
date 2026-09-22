"""Pull YouTube set-review transcripts and slice them into per-card commentary.

These reviewers talk about cards but almost never score them -- across five FRA
videos (61k words) there were three spoken ratings. So there is nothing to
scrape. What this does is cut the transcript into "what was said about card X",
so a human (or a model) can read the commentary and assign a number, with the
provenance of that number being honest about who produced it.

Auto-captions mangle proper nouns, so matching is fuzzy: exact name first, then
the most distinctive word in the name, then a difflib pass over word windows.
"""
import difflib, json, os, re, subprocess, sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "yt")
STOPWORDS = {"the", "of", "a", "an", "and", "to", "in"}


def fetch(video_id):
    os.makedirs(CACHE, exist_ok=True)
    vtt = os.path.join(CACHE, f"{video_id}.en.vtt")
    if not os.path.exists(vtt):
        subprocess.run([os.path.join(ROOT, ".venv/bin/yt-dlp"), "--skip-download",
                        "--write-auto-sub", "--sub-lang", "en.*", "--sub-format", "vtt",
                        "-o", os.path.join(CACHE, "%(id)s"),
                        f"https://www.youtube.com/watch?v={video_id}"],
                       capture_output=True)
    return vtt if os.path.exists(vtt) else None


def plain(vtt):
    lines = []
    for l in open(vtt, encoding="utf8", errors="replace"):
        if "-->" in l or l.startswith(("WEBVTT", "Kind:", "Language:")) or not l.strip():
            continue
        l = re.sub(r"<[^>]+>", "", l).strip()
        if l and (not lines or lines[-1] != l):
            lines.append(l)
    return re.sub(r"\s+", " ", " ".join(lines))


def key_word(name):
    """The most distinctive word in a card name -- survives ASR better than the whole."""
    words = [w for w in re.findall(r"[A-Za-z']{4,}", name.split(" // ")[0])
             if w.lower() not in STOPWORDS]
    return max(words, key=len).lower() if words else None


def find_mentions(text, cards, window=420):
    low = text.lower()
    words = low.split()
    out = {}
    for name in cards:
        front = name.split(" // ")[0]
        pos = low.find(front.lower())
        if pos < 0:
            kw = key_word(name)
            if kw and len(kw) >= 6:
                pos = low.find(kw)
            if pos < 0 and kw:
                # last resort: fuzzy over sliding word windows
                n = max(1, len(front.split()))
                best, bi = 0.0, -1
                for i in range(0, len(words) - n):
                    cand = " ".join(words[i:i + n])
                    r = difflib.SequenceMatcher(None, front.lower(), cand).ratio()
                    if r > best:
                        best, bi = r, i
                if best >= 0.78:
                    pos = low.find(words[bi], sum(len(w) + 1 for w in words[:bi]) - 2)
        if pos >= 0:
            out[name] = text[max(0, pos - 60): pos + window].strip()
    return out


if __name__ == "__main__":
    VIDEOS = {
        "gpUHzzmvqXw": "White", "sdNDgZVI7a0": "Blue", "NcjzcXS8KtA": "Black",
        "EZC9fwysE3o": "Red", "jtg04oa-k5o": "Green", "YJFq9pJ3LKk": "Multicolor",
        "VMIreWfgZuI": "Colorless", "iM6OaXGuh6o": "Multicolor/Colorless (2nd)",
        "pmt47XZUUZ4": "White (2nd)", "JhGxtY9WmDc": "North 100",
    }
    cards = [c["name"] for c in json.load(open(f"{ROOT}/data/cards/FRA.json"))]
    all_out = {}
    for vid, label in VIDEOS.items():
        vtt = fetch(vid)
        if not vtt:
            print(f"{label:28s} no captions"); continue
        txt = plain(vtt)
        m = find_mentions(txt, cards)
        all_out[vid] = {"label": label, "words": len(txt.split()), "cards": m}
        print(f"{label:28s} {len(txt.split()):6,d} words  {len(m):3d} cards matched")
    json.dump(all_out, open(f"{ROOT}/data/yt_mentions.json", "w"), indent=1)
    tot = len({c for v in all_out.values() for c in v["cards"]})
    print(f"\n{tot} distinct cards have commentary somewhere")
