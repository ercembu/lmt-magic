"""Scryfall's community oracle tags, per card.

Every feature in features.py is something we derived from the card ourselves.
These are not: they are a human-curated functional taxonomy, maintained by
Scryfall's tagger community, and they catch things a regex does not -- "evasion"
includes landwalk, shadow and protection-based evasion; "combat-trick" knows an
instant that pumps is a different object from one that draws.

The reason this is worth trying when a dozen hand-written feature families were
not: it is new information rather than a cleverer reading of information we
already had, and that is the only kind of change that has ever worked here.

Crucially the taggers are ahead of release -- TRK was already tagged a week
before its prerelease -- so this is available when it is needed.
"""
import json, os, sys, time, urllib.error, urllib.parse, urllib.request

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "otags.json")

TAGS = ["artifact-removal", "attack-trigger", "board-wipe", "burn", "cantrip",
        "card-advantage", "cast-trigger", "combat-trick", "counter-spell",
        "creature-removal", "death-trigger", "edict", "enchantment-removal",
        "evasion", "impulse-draw", "mana-ramp", "mass-removal",
        "planeswalker-removal", "ramp", "removal", "ritual", "scry",
        "spot-removal", "sweeper", "tutor", "unblockable"]


def get(url, tries=6):
    """Scryfall asks for 50-100ms between calls and 429s if you forget."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "lmt-magic/0.1 (limited card grader research)",
        "Accept": "application/json"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                time.sleep(5 * (i + 1))
                continue
            raise
        except urllib.error.URLError:
            if i < tries - 1:
                time.sleep(3 * (i + 1))
                continue
            raise


def search(query):
    cards, url = [], ("https://api.scryfall.com/cards/search?q=" +
                      urllib.parse.quote(query) + "&unique=cards")
    while url:
        try:
            d = get(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return cards
            raise
        cards.extend(d["data"])
        url = d.get("next_page")
        time.sleep(0.25)
    return cards


if __name__ == "__main__":
    sets = sorted(f[:-5] for f in os.listdir(f"{ROOT}/data/cards") if f.endswith(".json"))
    clause = "(" + " or ".join(f"set:{s.lower()}" for s in sets) + ")"
    out = {}
    for t in TAGS:
        hits = search(f"otag:{t} {clause}")
        time.sleep(0.5)
        for c in hits:
            for n in {c["name"], c["name"].split(" // ")[0]}:
                out.setdefault(n, []).append(t)
        print(f"  {t:22s} {len(hits):5d} cards", flush=True)
    json.dump(out, open(OUT, "w"))
    n_tags = sum(len(v) for v in out.values())
    print(f"\n{len(out)} cards carry {n_tags} tags across {len(sets)} sets")
    for s in ("FRA", "TRK"):
        cards = json.load(open(f"{ROOT}/data/cards/{s}.json"))
        hit = sum(1 for c in cards if c["name"] in out or c["name"].split(" // ")[0] in out)
        print(f"  {s}: {hit}/{len(cards)} cards tagged")
