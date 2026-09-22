"""What the set review actually argues, collected by idea instead of by card.

The review page lists 280 ratings. The reasoning beside them is ~275 short
written evaluations that say things the numbers cannot: that a mechanic is
weaker than it looks, that a card needs three others to work, that the format
is slower than the last one. Read card by card those observations are scattered
across 280 entries; collected by subject they are the closest thing to a
strategy article we have for a set nobody has played.

Everything here is extracted and quoted, never paraphrased into a new claim.
Each line keeps the card it came from so it can be checked.

One source: the prose is Draftsim's. MTG Arena Zone publishes ratings we can
parse but their write-ups are not in our cache, so this page is one person's
reasoning -- which is exactly why the ratings pages carry two.
"""
import html, json, os, re, sys
from collections import Counter, defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")

# FRA's own vocabulary. Empower is the set's build-around: each colour has a
# planeswalker token it puts loyalty on, and the five two-colour archetypes are
# named for those walkers.
MECHANICS = [
    ("Empower", r"\bempower(?:s|ed|ing)?\b",
     "Put loyalty counters on a planeswalker token, creating it first if you "
     "don't have one. Five walkers, one per colour, and the archetypes are named "
     "for them."),
    ("Prepared", r"\bprepare[ds]?\b",
     "A creature that enters prepared carries a spell on its back: while it is "
     "prepared you may cast a copy of that spell, which unprepares it. Two cards "
     "in one, and the reason the reviewer keeps calling these cheap."),
    ("Surveil", r"\bsurveil(?:s|ed|ing)?\b",
     "Look at the top card, bin it or keep it. Selection that also fills a "
     "graveyard, which is why it turns up in both the white-blue aggro deck and "
     "the blue-black graveyard one."),
    ("Legendary", r"\bLegendary\b",
     "A third of the set. Named characters, many appearing twice in different "
     "colours, which is what makes 'every deck has a legend in it' a safe "
     "assumption here and a payoff worth building on.", "type"),
]

# Subjects the reviewer returns to. A sentence is filed under every bucket whose
# pattern it matches, so a claim about slow removal appears under both.
THEMES = [
    ("Format speed", r"\b(?:slow|fast|aggro|aggressive|tempo|race|racing|grind|grindy|"
                     r"stall|clock|curve out|early|late game)\b",
     "How quickly games end, and whether beating down beats building up."),
    ("Removal", r"\b(?:removal|kill spell|answer[s]?\b|deal with|interaction|"
                r"trick[s]?\b|combat trick)\b",
     "How much interaction the set has and what it costs."),
    ("Bombs and rares", r"\b(?:bomb|bombs|game[- ]?ending|wins? the game|take[s]? over|"
                        r"unanswered|must[- ]answer|demands? an answer|first[- ]pick|"
                        r"win the game on the spot|run away with)\b",
     "Cards that decide a game if nobody deals with them."),
    ("Build-arounds", r"\b(?:build[- ]?around|payoff|enabler|support(?:ed|s)?\b|"
                      r"synerg\w+|needs? (?:a lot|some|the rest)|only (?:if|works))\b",
     "Cards that do nothing alone and a lot with help."),
    ("Commons and filler", r"\b(?:common|uncommon|filler|playable|unplayable|"
                           r"23rd card|last card|sideboard)\b",
     "The cards that actually make up the deck."),
    ("Evasion and stalls", r"\b(?:flying|flier[s]?\b|evasion|evasive|menace|trample|"
                           r"ground stall|board stall|gum up|break (?:a|the) stall)\b",
     "How a board that has stopped moving gets unstuck."),
]

SUPERLATIVE = re.compile(
    r"\b(?:best|strongest|worst|weakest|first[- ]pick|premium|10/10|0/10|"
    r"format[- ]warping|absurd|busted|insane|nuts|unplayable|never play)\b", re.I)


def clean(text):
    """Strip the image block that begins the next card's write-up."""
    t = re.split(r'<img|\bwidth="|\bstyle="', text)[0]
    t = re.sub(r"(?is)<[^>]*>", " ", t)          # stray anchors around card links
    t = t.split("<")[0]                          # cached prose can end mid-tag
    return html.unescape(re.sub(r"\s+", " ", t)).strip()


JUNK = re.compile(r'https?://|\bwp-content\b|\bclass="|\bdata-\w+=')


def sentences(text):
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip()
        if 30 <= len(s) <= 320 and not JUNK.search(s):
            out.append(s)
    return out


def load(setcode):
    prose = [p for p in json.load(open(f"{ROOT}/data/prose.json")) if p["set"] == setcode]
    grades = {g["name"]: g for g in json.load(open(f"{ROOT}/data/grades_{setcode}.json"))}
    for p in prose:
        p["text"] = clean(p["prose"])
        g = grades.get(p["name"]) or next(
            (v for k, v in grades.items() if k.split(" // ")[0] == p["name"]), None)
        p["grade"] = g["grade"] if g else None
        p["uri"] = g["scryfall_uri"] if g else None
        p["image"] = g.get("image") if g else None
        p["colors"] = (g.get("colors") or []) if g else []
        # The set's Special Guests are graded like everything else now, but they
        # are rare enough in Play Boosters that a page should say which is which.
        p["special"] = bool(g and g.get("bonus_sheet")) or g is None
    return prose, grades


def mechanics(prose, grades, setcode):
    """For each set mechanic: how many cards carry it, and what he says about it."""
    cards = json.load(open(f"{ROOT}/data/cards/{setcode}.json"))
    def otext(c):
        faces = c.get("card_faces", [])
        return " ".join([c.get("oracle_text") or "", c.get("type_line") or ""] +
                        [f.get("oracle_text", "") for f in faces] +
                        [f.get("type_line", "") for f in faces])
    out = []
    for entry in MECHANICS:
        name, pat, blurb = entry[:3]
        scope = entry[3] if len(entry) > 3 else "all"
        rx = re.compile(pat, re.I)
        field = (lambda c: (c.get("type_line") or "")) if scope == "type" else otext
        carriers = [c["name"] for c in cards if rx.search(field(c))]
        quotes = []
        for p in prose:
            for s in sentences(p["text"]):
                if rx.search(s):
                    quotes.append({"name": p["name"], "grade": p["grade"], "special": p["special"],
                                   "rating": p["rating"], "uri": p["uri"],
                                   "image": p["image"], "text": s})
        # the ones with an argument in them first: longer sentences that are not
        # simply restating the card's own rules text
        quotes.sort(key=lambda q: -(len(q["text"]) + 60 * bool(SUPERLATIVE.search(q["text"]))))
        best = sorted((c for c in carriers if c in grades),
                      key=lambda n: -grades[n]["z"])[:8]
        if carriers:
            out.append({"name": name, "blurb": blurb, "cards": len(carriers),
                        "top": [{"name": n, "grade": grades[n]["grade"],
                                 "uri": grades[n]["scryfall_uri"],
                                 "image": grades[n].get("image")} for n in best],
                        "quotes": quotes[:6]})
    return out


def themes(prose):
    out = []
    for name, pat, blurb in THEMES:
        rx = re.compile(pat, re.I)
        found, seen = [], set()
        for p in prose:
            for s in sentences(p["text"]):
                if not rx.search(s):
                    continue
                key = re.sub(r"[^a-z]", "", s.lower())[:60]
                if key in seen:
                    continue
                seen.add(key)
                found.append({"name": p["name"], "grade": p["grade"], "special": p["special"],
                              "rating": p["rating"], "uri": p["uri"], "text": s,
                              "image": p["image"],
                              "strong": bool(SUPERLATIVE.search(s))})
        found.sort(key=lambda q: (-q["strong"], -len(q["text"])))
        out.append({"name": name, "blurb": blurb, "count": len(found),
                    "quotes": found[:14]})
    return out


def comparisons(prose, setcode):
    """Older cards the reviewer reaches for to explain a new one.

    "A colour-shifted Displacer Kitten" tells you more about an unreleased card
    than any feature we extract, and he does it constantly. The name pool is
    every set we have cached, minus this one.
    """
    pool, art = {}, {}
    for fn in sorted(os.listdir(f"{ROOT}/data/cards")):
        code = fn[:-5]
        if code == setcode:
            continue
        for c in json.load(open(f"{ROOT}/data/cards/{fn}")):
            n = c["name"].split(" // ")[0]
            if len(n) >= 10 and " " in n:
                pool.setdefault(n, c.get("scryfall_uri"))
                art.setdefault(n, (c.get("image_uris")
                                   or (c.get("card_faces", [{}])[0].get("image_uris") or {})
                                   ).get("normal"))
    here = {g.split(" // ")[0] for g in
            json.load(open(f"{ROOT}/data/cards/{setcode}.json"))[0].keys()} if False else set()
    here = {c["name"].split(" // ")[0]
            for c in json.load(open(f"{ROOT}/data/cards/{setcode}.json"))}
    names = sorted((n for n in pool if n not in here), key=len, reverse=True)
    found = []
    for p in prose:
        text = p["text"]
        own = p["name"].split(" // ")[0]
        for n in names:
            if n not in text or n == own or n in own or own in n:
                continue
            sent = next((s for s in sentences(text) if n in s), None)
            if sent:
                found.append({"name": p["name"], "grade": p["grade"], "uri": p["uri"],
                              "special": p["special"], "image": p["image"],
                              "rating": p["rating"], "compared_to": n,
                              "compared_uri": pool[n], "compared_image": art.get(n),
                              "text": sent})
            break
    found.sort(key=lambda q: -q["rating"])
    return found


def synergies(prose, grades, setcode):
    """Which cards the reviewer names in *other* cards' write-ups.

    A write-up that reaches for another card by name is the reviewer stating a
    synergy outright. Short and one-word names are skipped -- "Loot" and "Pia"
    match too much ordinary prose to trust.
    """
    info = json.load(open(f"{ROOT}/data/setinfo/{setcode}.json"))
    names = [n for n in grades if len(n) >= 8 and " " in n.split(" // ")[0]]
    names.sort(key=len, reverse=True)
    edges = defaultdict(list)
    for p in prose:
        text = p["text"]
        for n in names:
            short = n.split(" // ")[0]
            if short == p["name"] or short not in text:
                continue
            for s in sentences(text):
                if short in s:
                    edges[short].append({"from": p["name"], "grade": p["grade"],
                                         "uri": p["uri"], "text": s})
                    break
    named = sorted(edges.items(), key=lambda kv: -len(kv[1]))
    named = [kv for kv in named if len(kv[1]) >= 2]
    # the archetypes by the names the reviewer uses for them
    EMPOWER = re.compile(r"\bempower(?:s|ed|ing)?\s+[A-Z][a-z]+\s*\d*", re.I)
    # Five archetypes are named for the planeswalker tokens, and those tokens are
    # also a keyword on 35 cards. "The Jace tokens are good here" is a remark
    # about the mechanic; only a possessive or an explicit "Jace deck" is a remark
    # about the archetype. The other five have proper nouns of their own and need
    # no such care.
    WALKERS = {"Jace", "Ajani", "Chandra", "Garruk", "Liliana"}
    arch = []
    for pair, a in info["archetypes"].items():
        word = a["name"].split()[0].rstrip("'s").rstrip("\u2019s")
        rx = (re.compile(r"\b" + re.escape(word) +
                         r"(?:'s|\u2019s\b|\s+(?:deck|build|archetype|pair|side))", re.I)
              if word in WALKERS else re.compile(r"\b" + re.escape(word) + r"\b", re.I))
        hits, seen = [], set()
        for p in prose:
            for s in sentences(EMPOWER.sub(" ", p["text"])):
                if rx.search(s) and s not in seen:
                    seen.add(s)
                    hits.append({"name": p["name"], "grade": p["grade"], "special": p["special"],
                                 "uri": p["uri"], "image": p["image"], "text": s})
        hits.sort(key=lambda q: -len(q["text"]))
        arch.append({"pair": pair, "name": a["name"], "hook": a.get("hook", ""),
                     "word": word, "count": len(hits), "quotes": hits[:8]})
    arch.sort(key=lambda a: -a["count"])
    return {
        "most_named": [{"name": n, "grade": grades[n]["grade"] if n in grades else None,
                        "uri": grades[n]["scryfall_uri"] if n in grades else None,
                        "count": len(v), "mentions": v[:5]}
                       for n, v in named[:14]],
        "archetypes": arch,
    }


def verdicts(prose):
    """The lines where he commits to something, high or low."""
    hi, lo = [], []
    for p in prose:
        for s in sentences(p["text"]):
            if not SUPERLATIVE.search(s):
                continue
            rec = {"name": p["name"], "grade": p["grade"], "rating": p["rating"],
                   "uri": p["uri"], "image": p["image"], "text": s,
                   "special": p["special"]}
            (hi if p["rating"] >= 6 else lo if p["rating"] <= 2.5 else []).append(rec)
    def dedupe(rows):
        seen, out = set(), []
        for r in rows:                       # one line per card, the longest one
            if r["name"] in seen:
                continue
            seen.add(r["name"])
            out.append(r)
        return out
    hi.sort(key=lambda q: (-q["rating"], -len(q["text"])))
    lo.sort(key=lambda q: (q["rating"], -len(q["text"])))
    return {"high": dedupe(hi)[:12], "low": dedupe(lo)[:12]}


def build(setcode="FRA"):
    prose, grades = load(setcode)
    words = sum(len(p["text"].split()) for p in prose)
    special = sorted({p["name"] for p in prose if p["special"]})
    return {"set": setcode, "cards": len(prose), "words": words,
            "special_guests": special,
            "source": {"name": "Draftsim", "note": "ratings 0-10"},
            "mechanics": mechanics(prose, grades, setcode),
            "themes": themes(prose),
            "synergies": synergies(prose, grades, setcode),
            "comparisons": comparisons(prose, setcode),
            "verdicts": verdicts(prose)}


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    d = build(code)
    json.dump(d, open(f"{ROOT}/data/ideas_{code}.json", "w"), indent=1)
    print(f"{code}: {d['cards']} write-ups, {d['words']:,} words")
    print(f"special guests (graded, but left out of pool statistics): "
          f"{len(d['special_guests'])} -- {', '.join(d['special_guests'])}")
    print("\nmechanics:")
    for m in d["mechanics"]:
        print(f"  {m['name']:16s} {m['cards']:3d} cards  {len(m['quotes'])} quotes")
    print("\nthemes:")
    for t in d["themes"]:
        print(f"  {t['name']:20s} {t['count']:3d} sentences")
    print(f"\nold cards he reaches for: {len(d['comparisons'])}")
    for c in d["comparisons"][:8]:
        print(f"  {c['name'][:28]:30s} -> {c['compared_to']}")
    print("\nmost-named cards in other write-ups:")
    for m in d["synergies"]["most_named"][:8]:
        print(f"  {m['name'][:34]:36s} {m['count']:2d}")
    print("\narchetypes by how much ink they get:")
    for a in d["synergies"]["archetypes"]:
        print(f"  {a['pair']}  {a['name'][:30]:32s} {a['count']:3d}")
