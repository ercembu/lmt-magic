"""Second reviewer: MTG Arena Zone (j2sjosh), 0.0-5.0 scale.

Their site sits behind a JavaScript proof-of-work challenge, so curl and a
headless Firefox both get "Checking your browser..." instead of the article.
The fetch therefore has to come from something that runs JS; the parsing and
merging below is scripted so that step is the only manual one.

Drop one file per colour into data/reviews_mtgazone/ named <SET>-<colour>.txt,
each line "Card Name — rating". Comment lines starting with # are ignored.
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
from expert_grades import norm

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "data", "reviews_mtgazone")
OUT = os.path.join(ROOT, "data", "expert_mtgazone")
SCALE_MAX = 5.0


def parse_file(path):
    out = {}
    for line in open(path, encoding="utf8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # "Card Name — 3.5", also tolerates -, –, :
        m = re.match(r"^(.*?)\s*[—–\-:]\s*([0-5](?:\.\d)?)\s*$", line)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
    return out


def resolve(ratings, setcode):
    cards = {c["name"] for c in json.load(open(f"{ROOT}/data/cards/{setcode}.json"))}
    # full names before front faces before names cut at the comma, so "Seven of Nine" isn't taken by
    # "Seven of Nine, Unassimilated"; norm() evens out curly quotes and ". . ."
    idx = {}
    for alias_of in (lambda k: k, lambda k: k.split(" // ")[0], lambda k: k.split(",")[0]):
        for k in sorted(cards):
            idx.setdefault(alias_of(k), k)
            idx.setdefault(norm(alias_of(k)), k)
    hit, miss = {}, []
    for n, r in ratings.items():
        k = idx.get(n) or idx.get(norm(n))
        if k:
            hit[k] = r
        else:
            miss.append(n)
    return hit, miss


def load(setcode):
    """Everything published so far for this set, already resolved."""
    if not os.path.isdir(SRC):
        return {}, []
    all_r, all_miss = {}, []
    for f in sorted(os.listdir(SRC)):
        if not f.startswith(f"{setcode}-") or not f.endswith(".txt"):
            continue
        hit, miss = resolve(parse_file(os.path.join(SRC, f)), setcode)
        all_r.update(hit)
        all_miss += miss
    return all_r, all_miss


if __name__ == "__main__":
    setcode = sys.argv[1] if len(sys.argv) > 1 else "TRK"
    ratings, miss = load(setcode)
    if not ratings:
        raise SystemExit(f"no {setcode}-*.txt files in {SRC}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(ratings, open(f"{OUT}/{setcode}.json", "w"), indent=1)
    parts = sorted(f.split("-", 1)[1][:-4] for f in os.listdir(SRC)
                   if f.startswith(f"{setcode}-") and f.endswith(".txt"))
    total = len(json.load(open(f"{ROOT}/data/cards/{setcode}.json")))
    print(f"{setcode}: {len(ratings)} cards rated across {len(parts)} part(s): {', '.join(parts)}")
    print(f"  covers {len(ratings)}/{total} cards in the set")
    if miss:
        print(f"  {len(miss)} names did not resolve: {miss[:6]}")
