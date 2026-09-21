"""Pull the reviewer's written reasoning out of the cached set reviews.

expert_grades.py takes only the number. The prose beside it is a few thousand
short explanations of *why*, which is the one place a human states the things
card text cannot: that ten mana is untenable, that a card only works with
support, that a body trades up. Useful for finding what the model is blind to.
"""
import html, json, os, re, sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "reviews")
OUT = os.path.join(ROOT, "data", "prose.json")


def parse(html_text, setcode):
    dec = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), html_text)
    parts = re.split(r"Rating:\s*(?:</?\w+[^>]*>\s*)*(\d+(?:\.\d)?)\s*/\s*10", dec)
    out = []
    for i in range(1, len(parts), 2):
        names = re.findall(r'alt="([^"]{2,70})"', parts[i - 1])
        if not names:
            continue
        body = re.sub(r"(?is)<[^>]+>", " ", parts[i + 1][:1800])
        body = html.unescape(re.sub(r"\s+", " ", body)).strip()
        # the next card's image block starts the next write-up; cut there
        body = re.split(r"\b\d{3,4}_MTG", body)[0].strip()
        if len(body) > 40:
            out.append({"set": setcode, "name": names[-1].strip(),
                        "rating": float(parts[i]), "prose": body[:900]})
    return out


if __name__ == "__main__":
    recs = []
    for f in sorted(os.listdir(CACHE)):
        if not f.endswith(".html"):
            continue
        code = f[:-5]
        recs += parse(open(os.path.join(CACHE, f), encoding="utf8", errors="replace").read(), code)
    json.dump(recs, open(OUT, "w"))
    sets = {r["set"] for r in recs}
    words = sum(len(r["prose"].split()) for r in recs)
    print(f"{len(recs):,} write-ups from {len(sets)} sets  (~{words:,} words)")
    print("sets:", " ".join(sorted(sets)))
