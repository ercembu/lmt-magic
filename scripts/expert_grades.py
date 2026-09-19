"""Scrape published pre-release set reviews for expert card ratings.

This is the only signal in the project that is not derived from card text: a
human reading a card sees synergy and format speed that no feature here
captures. Reviews are published before release, so unlike win rates they are
available for the set we actually want to grade.

Draftsim publishes one review per set at /mtg-<code>-limited-set-review/ with
each card as an image (alt = card name) followed by "Rating: N/10".
"""
import json, os, re, subprocess, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "reviews")
OUT = os.path.join(ROOT, "data", "expert")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140.0 Safari/537.36"


def fetch(setcode, force=False):
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, f"{setcode}.html")
    if os.path.exists(dst) and not force and os.path.getsize(dst) > 200_000:
        return open(dst, encoding="utf8", errors="replace").read()
    url = f"https://draftsim.com/mtg-{setcode.lower()}-limited-set-review/"
    code = subprocess.run(["curl", "-s", "-L", "-m", "60", "-H", f"User-Agent: {UA}",
                           "-o", dst, "-w", "%{http_code}", url],
                          capture_output=True, text=True).stdout.strip()
    if code != "200":
        if os.path.exists(dst):
            os.remove(dst)
        return None
    return open(dst, encoding="utf8", errors="replace").read()


def parse(html):
    """Pair each 'Rating: N/10' with the nearest card image before it."""
    if not html:
        return {}
    marks = []
    for m in re.finditer(r'alt="([^"]{2,80})"', html):
        marks.append(("name", m.start(), m.group(1)))
    # tags may or may not sit between "Rating:" and the number, so the tag group
    # has to be optional as a whole -- "</?\w*>?" still requires the "<"
    for m in re.finditer(r"Rating:\s*(?:</?\w+[^>]*>\s*)*(\d+(?:\.\d)?)\s*/\s*10", html):
        marks.append(("rating", m.start(), float(m.group(1))))
    marks.sort(key=lambda x: x[1])
    out, last = {}, None
    for kind, _, val in marks:
        if kind == "name":
            last = val
        elif last is not None:
            out.setdefault(last, val)      # first rating after a card wins
            last = None
    return out


def resolve(ratings, setcode):
    """Keep only names that are real cards in this set."""
    p = f"{ROOT}/data/cards/{setcode}.json"
    if not os.path.exists(p):
        return {}
    cards = {c["name"] for c in json.load(open(p))}
    idx = {}
    for k in cards:
        idx[k] = k
        idx[k.split(" // ")[0]] = k
        idx[k.split(",")[0]] = k
    return {idx[n]: r for n, r in ratings.items() if n in idx}


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    import train
    os.makedirs(OUT, exist_ok=True)
    want = sys.argv[1:] or (train.TRAIN_SETS + ["FRA"])
    total = 0
    for s in want:
        html = fetch(s)
        if html is None:
            print(f"{s:5s} no review published"); continue
        got = resolve(parse(html), s)
        if len(got) < 40:
            print(f"{s:5s} parsed only {len(got)} cards — skipping"); continue
        json.dump(got, open(f"{OUT}/{s}.json", "w"), indent=1)
        vals = list(got.values())
        print(f"{s:5s} {len(got):3d} cards rated   mean {np.mean(vals):.2f}  "
              f"range {min(vals):.1f}-{max(vals):.1f}")
        total += 1
    print(f"\nreviews usable for {total} sets")
