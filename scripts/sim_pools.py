"""Random sealed pools for a set, built into 40-card decks for XMage to play.

Simulated games are the only way to get anything like a win rate for a set
before 17lands has one. This makes the decks; xmage_sim plays them; sim_fit.py
turns the results into a per-card value.

Two choices that decide whether the simulation can be trusted:

  * Decks are built from the published review consensus, plus noise -- never
    from our own grades. If the model chose the decks, the cards it dislikes
    would never be played and the simulation could only echo the model back.
  * The noise is large on purpose, so marginal cards get played often enough to
    be measured. A card that is never in a deck has no simulated value at all.

Packs approximate a Play Booster: 7 commons, 3 uncommons, a rare (1 in 8
mythic) and a wildcard. Bonus sheets are ignored.
"""
import json, os, random, sys
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), "..")
COLOURS = "WUBRG"
BASIC = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}


def xmage_names(path):
    """Card names XMage implements for a set, read from its set class. Anything
    else is dropped by XMage's deck importer without complaint."""
    import re
    return set(re.findall(r'cards\.add\(new SetCardInfo\("([^"]+)"', open(path).read()))


def load(setcode, implemented=None):
    cards = [c for c in json.load(open(f"{ROOT}/data/cards/{setcode}.json"))
             if not c.get("bonus_sheet") and "Basic Land" not in c.get("type_line", "")]
    # one printing per name: the main-set one with the lowest collector number
    by = {}
    for c in cards:
        k = c["name"]
        n = int("".join(ch for ch in c["collector_number"] if ch.isdigit()) or 9999)
        if k not in by or n < by[k][0]:
            by[k] = (n, c)
    cards = [c for _, c in by.values()]
    if implemented is not None:
        cards = [c for c in cards if c["name"].split(" // ")[0] in implemented]
    ep = f"{ROOT}/data/expert_consensus/{setcode}.json"
    if not os.path.exists(ep):
        ep = f"{ROOT}/data/expert/{setcode}.json"
    rating = json.load(open(ep)) if os.path.exists(ep) else {}
    return cards, rating


def open_pack(rng, by_rarity):
    p = rng.sample(by_rarity["common"], 7) + rng.sample(by_rarity["uncommon"], 3)
    p.append(rng.choice(by_rarity["mythic"] if rng.random() < 1 / 8 and by_rarity["mythic"]
                        else by_rarity["rare"]))
    r = rng.random()
    tier = "common" if r < 0.55 else "uncommon" if r < 0.85 else "rare" if r < 0.97 else "mythic"
    p.append(rng.choice(by_rarity[tier] or by_rarity["rare"]))
    return p


def build(pool, rating, rng, noise=2.0):
    """Best two-colour 23 + 17 basics, judged by the review with noise added."""
    score = {id(c): rating.get(c["name"], 2.0) + rng.gauss(0, noise) for c in pool}
    spells = [c for c in pool if "Land" not in c.get("type_line", "")]
    best = None
    for i, a in enumerate(COLOURS):
        for b in COLOURS[i + 1:]:
            ok = [c for c in spells if set(c.get("color_identity") or []) <= {a, b}]
            ok.sort(key=lambda c: -score[id(c)])
            crea = [c for c in ok if "Creature" in c["type_line"]]
            rest = [c for c in ok if "Creature" not in c["type_line"]]
            # at least 13 creatures where the pool allows it
            deck = crea[:13]
            deck += sorted([c for c in ok if c not in deck], key=lambda c: -score[id(c)])[:23 - len(deck)]
            # a thin pair runs extra lands rather than off-colour cards
            if len(deck) < 20:
                continue
            total = sum(score[id(c)] for c in deck)
            total += 0.5 * (len(deck) - 23)          # prefer full 23-spell builds
            if best is None or total > best[0]:
                best = (total, a + b, deck)
    _, pair, deck = best
    pips = Counter()
    for c in deck:
        for col in pair:
            pips[col] += (c.get("mana_cost") or "").count("{" + col + "}")
    tot = sum(pips.values()) or 1
    n_lands = 40 - len(deck)
    na = max(6, min(n_lands - 6, round(n_lands * pips[pair[0]] / tot)))
    lands = {BASIC[pair[0]]: na, BASIC[pair[1]]: n_lands - na}
    return pair, deck, lands


def basics(setcode):
    """The set's own basic land printings -- XMage cannot resolve a bare name."""
    return {c["name"]: c["collector_number"]
            for c in json.load(open(f"{ROOT}/data/cards/{setcode}.json"))
            if "Basic Land" in c.get("type_line", "") and c["name"] in BASIC.values()}


def dck(deck, lands, setcode):
    n = Counter((c["collector_number"], c["name"]) for c in deck)
    # XMage names a double-faced or split card by its front face
    lines = [f"{k} [{setcode}:{num}] {name.split(' // ')[0]}" for (num, name), k in n.items()]
    num = basics(setcode)
    lines += [f"{n} [{setcode}:{num[name]}] {name}" for name, n in lands.items() if n]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    setcode = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    out = sys.argv[3] if len(sys.argv) > 3 else f"{ROOT}/data/sim/{setcode}/decks"
    xset = sys.argv[4] if len(sys.argv) > 4 else None     # XMage's Mage.Sets/.../<Set>.java
    os.makedirs(out, exist_ok=True)
    cards, rating = load(setcode, xmage_names(xset) if xset else None)
    by_rarity = {r: [c for c in cards if c["rarity"] == r]
                 for r in ("common", "uncommon", "rare", "mythic")}
    print(f"{setcode}: {len(cards)} booster cards, "
          f"{sum(len(v) for v in by_rarity.values())} by rarity, review covers "
          f"{sum(1 for c in cards if c['name'] in rating)}")
    rng = random.Random(0)
    manifest, seen = [], Counter()
    for i in range(n):
        pool = [c for _ in range(6) for c in open_pack(rng, by_rarity)]
        pair, deck, lands = build(pool, rating, rng)
        name = f"{setcode}_{i:05d}"
        open(f"{out}/{name}.dck", "w").write(dck(deck, lands, setcode))
        manifest.append({"deck": name, "pair": pair, "cards": [c["name"] for c in deck]})
        seen.update(c["name"] for c in deck)
    json.dump(manifest, open(f"{out}/manifest.json", "w"))
    played = len(seen)
    print(f"{n} decks; {played}/{len(cards)} cards appear in at least one; "
          f"median appearances {sorted(seen.values())[len(seen) // 2]}")
    print(Counter(m["pair"] for m in manifest).most_common())
