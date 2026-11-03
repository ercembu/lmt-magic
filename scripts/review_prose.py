"""Pull the reviewer's written reasoning out of the cached set reviews.

expert_grades.py takes only the number. The prose beside it is a few thousand
short explanations of *why*, which is the one place a human states the things
card text cannot: that ten mana is untenable, that a card only works with
support, that a body trades up. Useful for finding what the model is blind to.
"""
import html, json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
import expert_grades

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "reviews")
OUT = os.path.join(ROOT, "data", "prose.json")


def parse(html_text, setcode):
    dec = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), html_text)
    parts = re.split(r"Rating:\s*(?:</?\w+[^>]*>\s*)*(\d+(?:\.\d)?)\s*/\s*10", dec)
    out = []
    for i in range(1, len(parts), 2):
        # the card is the last name before its rating: an image's alt text, or the section heading
        # for cards shown without one (TRK's "All Good Things . . .") or with an empty alt
        marks = [(m.start(), m.group(1)) for m in re.finditer(r'alt="([^"]{2,70})"', parts[i - 1])]
        marks += [(m.start(), html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip())
                  for m in re.finditer(r"<h3[^>]*>(.*?)</h3>", parts[i - 1], re.S)]
        marks = [mk for mk in marks if 2 <= len(mk[1]) <= 70]
        if not marks:
            continue
        name = max(marks)[1]
        # the next card's heading or image block starts the next write-up; cut there
        raw = re.split(r"<h[23][\s>]|<div class=\"card_gallery", parts[i + 1][:2500])[0]
        body = re.sub(r"(?is)<[^>]+>", " ", raw)
        body = html.unescape(re.sub(r"\s+", " ", body)).strip()
        body = re.split(r"\b\d{3,4}_MTG", body)[0].strip()
        if len(body) > 40:
            out.append({"set": setcode, "name": name,
                        "rating": float(parts[i]), "prose": body[:900]})
    return out


if __name__ == "__main__":
    recs = []
    for f in sorted(os.listdir(CACHE)):
        if not f.endswith(".html"):
            continue
        code = f[:-5]
        look = expert_grades.name_index(code)
        for r in parse(open(os.path.join(CACHE, f), encoding="utf8", errors="replace").read(), code):
            r["name"] = look(r["name"]) or r["name"]     # the set's spelling, so pages can look cards up
            recs.append(r)
    json.dump(recs, open(OUT, "w"))
    sets = {r["set"] for r in recs}
    words = sum(len(r["prose"].split()) for r in recs)
    print(f"{len(recs):,} write-ups from {len(sets)} sets  (~{words:,} words)")
    print("sets:", " ".join(sorted(sets)))
