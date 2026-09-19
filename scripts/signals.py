"""Which cards should make you commit to a colour?

Sealed is mostly one decision: pick two colours, then play the best 23 cards in
them. This turns the graded set into that decision.

Three things get computed:

  colour depth   -- how deep each colour is. If green has nine B-or-better
                    playables and blue has three, that is a standing reason to
                    lean green before you open anything.
  colour pulls   -- the single-colour cards strong enough to be a reason on their
                    own. A mono-coloured bomb commits one colour and leaves the
                    other open, which is why it is listed separately from gold.
  pair strength  -- for each two-colour pair, the quality of the best 23 cards
                    castable in it. This is the format's prior, not your pool:
                    it says which pairs reward you when the packs cooperate.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import guards

ROOT = os.path.join(os.path.dirname(__file__), "..")
WUBRG = "WUBRG"
COLOR_NAMES = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}
PAIRS = [a + b for i, a in enumerate(WUBRG) for b in WUBRG[i + 1:]]
DECK_SPELLS = 23          # a 40-card sealed deck is ~23 spells + 17 lands


def castable_in(card, pair):
    """Playable in a deck of exactly these two colours (colourless counts)."""
    ci = set(card.get("color_identity") or [])
    return ci <= set(pair)


def is_land(card):
    return "Land" in (card.get("type_line") or "")


def build(setcode="FRA"):
    cards = json.load(open(f"{ROOT}/data/grades_{setcode}.json"))
    info = guards.load_setinfo(setcode) or {"archetypes": {}, "removal": []}
    playable = [c for c in cards if not is_land(c)]

    # --- colour depth -------------------------------------------------------
    depth = {}
    for col in WUBRG:
        mine = [c for c in playable if col in (c.get("colors") or [])]
        mono = [c for c in mine if len(c.get("colors") or []) == 1]
        zs = sorted((c["z"] for c in mine), reverse=True)
        depth[col] = {
            "color": col, "name": COLOR_NAMES[col],
            "cards": len(mine), "mono": len(mono),
            "a_tier": sum(1 for c in mine if c["grade"][0] == "A"),
            "b_or_better": sum(1 for c in mine if c["grade"][0] in "AB"),
            "top15_mean_z": float(np.mean(zs[:15])) if zs else 0.0,
        }
    # Rank by the quality of the cards you would actually play. In sealed you run
    # ~15 cards of your deeper colour, so the mean grade of a colour's best 15 is
    # the decision-relevant figure -- a colour full of marginal playables and no
    # bombs is a worse place to be than a slightly thinner one with real top end.
    # z is measured against the set average, so 0 is a meaningful origin and the
    # bar can be drawn proportionally from it.
    order = sorted(depth.values(), key=lambda d: -d["top15_mean_z"])
    for i, d in enumerate(order):
        depth[d["color"]]["rank"] = i + 1

    # --- colour pulls: single-colour cards worth committing for -------------
    pulls = {}
    for col in WUBRG:
        mono = [c for c in playable if (c.get("colors") or []) == [col]]
        mono.sort(key=lambda c: -c["z"])
        pulls[col] = mono[:10]

    # --- pair strength + what is exclusive to each pair ---------------------
    pairs = {}
    for pair in PAIRS:
        pool = [c for c in playable if castable_in(c, pair)]
        pool.sort(key=lambda c: -c["z"])
        best = pool[:DECK_SPELLS]
        gold = [c for c in pool if set(c.get("colors") or []) == set(pair)]
        gold.sort(key=lambda c: -c["z"])
        arch = info["archetypes"].get(pair)
        keys = []
        if arch:
            byname = {c["name"]: c for c in cards}
            for n in arch["key_cards"]:
                hit = byname.get(n) or next(
                    (v for k, v in byname.items()
                     if k.split(" // ")[0] == n or k.split(",")[0] == n), None)
                if hit:
                    keys.append(hit)
            keys.sort(key=lambda c: -c["z"])
        pairs[pair] = {
            "pair": pair,
            "name": arch["name"] if arch else f"{COLOR_NAMES[pair[0]]}-{COLOR_NAMES[pair[1]]}",
            "hook": arch.get("hook", "") if arch else "",
            "deck_mean_z": float(np.mean([c["z"] for c in best])) if best else 0.0,
            "pool_size": len(pool),
            "gold": gold[:6],
            "key_cards": keys[:6],
            "top": best[:8],
        }
    ranked = sorted(pairs.values(), key=lambda p: -p["deck_mean_z"])
    for i, p in enumerate(ranked):
        pairs[p["pair"]]["rank"] = i + 1

    return {"set": setcode, "depth": depth, "pulls": pulls, "pairs": pairs,
            "colors": list(WUBRG), "color_names": COLOR_NAMES}


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    res = build(code)
    slim = lambda c: {k: c[k] for k in
                      ("name", "grade", "z", "rarity", "mana_cost", "type_line",
                       "image", "scryfall_uri", "colors", "confidence", "trust",
                       "oracle_text")}
    out = {"set": code, "color_names": COLOR_NAMES, "colors": list(WUBRG),
           "depth": res["depth"],
           "pulls": {k: [slim(c) for c in v] for k, v in res["pulls"].items()},
           "pairs": {k: {**{kk: vv for kk, vv in v.items()
                            if kk not in ("gold", "key_cards", "top")},
                         "gold": [slim(c) for c in v["gold"]],
                         "key_cards": [slim(c) for c in v["key_cards"]],
                         "top": [slim(c) for c in v["top"]]}
                     for k, v in res["pairs"].items()}}
    json.dump(out, open(f"{ROOT}/data/signals_{code}.json", "w"), indent=1)

    print(f"{code}: colour depth (deepest first)")
    for d in sorted(res["depth"].values(), key=lambda d: d["rank"]):
        print(f"  {d['rank']}. {d['name']:6s} {d['cards']:3d} cards  "
              f"{d['a_tier']:2d} A-tier  {d['b_or_better']:2d} B+  "
              f"top15 mean z {d['top15_mean_z']:+.2f}")
    print(f"\n{code}: pair strength (best {DECK_SPELLS} castable cards)")
    for p in sorted(res["pairs"].values(), key=lambda p: p["rank"]):
        print(f"  {p['rank']:2d}. {p['pair']}  {p['name'][:32]:34s} "
              f"deck mean z {p['deck_mean_z']:+.2f}  ({len(p['gold'])} gold)")
