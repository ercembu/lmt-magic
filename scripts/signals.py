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

Each of those is computed three times, because one estimate of "green is deep"
is worth much less than two that agree:

  model    the shipped grades
  blind    the same model with the expert rating withheld -- text and past win
           rates only
  review   the two-reviewer consensus, no model at all

Model and review are not independent (the model is *given* the consensus), which
is exactly why blind is here: blind vs review is the honest second opinion, and
the page reports how far apart they actually are rather than implying agreement.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import guards
from scipy.stats import spearmanr, norm

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


def in_pool(card):
    """Cards you can reasonably expect to open.

    Bonus-sheet cards (Special Guests) are in Play Boosters but most sealed pools
    will not contain one, so counting them toward a colour's depth or a pair's
    best 23 would describe a pool nobody gets. They are still graded and still on
    the card list -- they are just not part of the prior this page is about.
    """
    return not card.get("bonus_sheet")


VIEWS = ["model", "blind", "review"]
VIEW_LABEL = {"model": "the grades", "blind": "model, reviews withheld",
              "review": "the reviewers"}


def _rank_z(values):
    """Rank-normalise to z, so three different scales become comparable."""
    v = np.asarray(values, float)
    r = np.argsort(np.argsort(v)) + 1
    return norm.ppf(r / (len(v) + 1.0))


def alt_scores(setcode, cards):
    """Per-card z under the blind model and under the reviewers.

    Both come from the review page's own build, which already withholds the
    expert feature to get a blind number. Cards no reviewer covered are dropped
    from the review view rather than guessed at -- a colour is judged on the
    cards that view can actually see.
    """
    path = f"{ROOT}/data/reviewdata_{setcode}.json"
    if not os.path.exists(path):
        return {}
    recs = json.load(open(path))["cards"]
    blind = {r["name"]: r["blind_z"] for r in recs if r.get("blind_z") is not None}
    rated = [r for r in recs if r.get("consensus") is not None]
    review = dict(zip((r["name"] for r in rated),
                      _rank_z([r["consensus"] for r in rated])))
    return {"blind": blind, "review": review}


def build(setcode="FRA"):
    cards = json.load(open(f"{ROOT}/data/grades_{setcode}.json"))
    info = guards.load_setinfo(setcode) or {"archetypes": {}, "removal": []}
    playable = [c for c in cards if not is_land(c) and in_pool(c)]
    alt = alt_scores(setcode, cards)
    scores = {"model": {c["name"]: c["z"] for c in cards}, **alt}

    def score_of(view, c):
        return scores.get(view, {}).get(c["name"])

    def mean_top(view, pool, n):
        zs = sorted((z for z in (score_of(view, c) for c in pool)
                     if z is not None), reverse=True)[:n]
        return float(np.mean(zs)) if zs else None

    def ranked(view, groups, n):
        """{key: (score, rank)} for one view, rank 1 = best."""
        sc = {k: mean_top(view, pool, n) for k, pool in groups.items()}
        order = sorted((k for k in sc if sc[k] is not None), key=lambda k: -sc[k])
        return {k: {"score": sc[k], "rank": order.index(k) + 1 if sc[k] is not None else None}
                for k in sc}

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
    # the same question asked of each view
    col_pools = {col: [c for c in playable if col in (c.get("colors") or [])]
                 for col in WUBRG}
    dv = {v: ranked(v, col_pools, 15) for v in VIEWS if v == "model" or v in alt}
    for col in WUBRG:
        depth[col]["views"] = {v: dv[v][col] for v in dv}
        rs = [dv[v][col]["rank"] for v in dv if dv[v][col]["rank"]]
        depth[col]["rank_spread"] = (max(rs) - min(rs)) if len(rs) > 1 else 0

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
            byname = {c["name"]: c for c in cards if in_pool(c)}
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
    ord_pairs = sorted(pairs.values(), key=lambda p: -p["deck_mean_z"])
    for i, p in enumerate(ord_pairs):
        pairs[p["pair"]]["rank"] = i + 1
    pair_pools = {pr: [c for c in playable if castable_in(c, pr)] for pr in PAIRS}
    pv = {v: ranked(v, pair_pools, DECK_SPELLS) for v in VIEWS if v == "model" or v in alt}
    for pr in PAIRS:
        pairs[pr]["views"] = {v: pv[v][pr] for v in pv}
        rs = [pv[v][pr]["rank"] for v in pv if pv[v][pr]["rank"]]
        pairs[pr]["rank_spread"] = (max(rs) - min(rs)) if len(rs) > 1 else 0

    # How much the two opinions actually are two opinions. The model is handed
    # the consensus as a feature, so model-vs-review agreement is partly
    # circular; blind-vs-review is the number that means something.
    corr = {}
    if alt:
        for a, b in (("model", "review"), ("blind", "review"), ("model", "blind")):
            common = [c["name"] for c in playable
                      if score_of(a, c) is not None and score_of(b, c) is not None]
            if len(common) >= 30:
                corr[f"{a}_{b}"] = float(spearmanr(
                    [scores[a][n] for n in common], [scores[b][n] for n in common]).statistic)

    return {"set": setcode, "depth": depth, "pulls": pulls, "pairs": pairs,
            "colors": list(WUBRG), "color_names": COLOR_NAMES,
            "views": [v for v in VIEWS if v == "model" or v in alt],
            "view_labels": VIEW_LABEL, "agreement": corr}


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    res = build(code)
    slim = lambda c: {k: c[k] for k in
                      ("name", "grade", "z", "rarity", "mana_cost", "type_line",
                       "image", "scryfall_uri", "colors", "confidence", "trust",
                       "oracle_text")}
    out = {"set": code, "color_names": COLOR_NAMES, "colors": list(WUBRG),
           "views": res["views"], "view_labels": res["view_labels"],
           "agreement": res["agreement"],
           "depth": res["depth"],
           "pulls": {k: [slim(c) for c in v] for k, v in res["pulls"].items()},
           "pairs": {k: {**{kk: vv for kk, vv in v.items()
                            if kk not in ("gold", "key_cards", "top")},
                         "gold": [slim(c) for c in v["gold"]],
                         "key_cards": [slim(c) for c in v["key_cards"]],
                         "top": [slim(c) for c in v["top"]]}
                     for k, v in res["pairs"].items()}}
    json.dump(out, open(f"{ROOT}/data/signals_{code}.json", "w"), indent=1)

    nb = sum(1 for c in json.load(open(f"{ROOT}/data/grades_{code}.json"))
             if c.get("bonus_sheet"))
    if nb:
        print(f"({nb} bonus-sheet cards excluded -- openable, but not in a pool "
              f"you can plan around)\n")
    print(f"{code}: colour depth (deepest first)")
    for d in sorted(res["depth"].values(), key=lambda d: d["rank"]):
        print(f"  {d['rank']}. {d['name']:6s} {d['cards']:3d} cards  "
              f"{d['a_tier']:2d} A-tier  {d['b_or_better']:2d} B+  "
              f"top15 mean z {d['top15_mean_z']:+.2f}")
    print(f"\n{code}: pair strength (best {DECK_SPELLS} castable cards)")
    for p in sorted(res["pairs"].values(), key=lambda p: p["rank"]):
        rr = "  ".join(f"{v}:{p['views'][v]['rank']}" for v in res["views"])
        print(f"  {p['rank']:2d}. {p['pair']}  {p['name'][:32]:34s} "
              f"deck mean z {p['deck_mean_z']:+.2f}  ({len(p['gold'])} gold)  [{rr}]")
    if res["agreement"]:
        print("\nhow much the views agree (spearman over the set):")
        for k, v in res["agreement"].items():
            print(f"  {k.replace('_',' vs '):22s} {v:+.3f}")
        print("\ncolour depth by view (rank 1 = deepest):")
        for col in WUBRG:
            d = res["depth"][col]
            rr = "  ".join(f"{v}:{d['views'][v]['rank']}" for v in res["views"])
            print(f"  {COLOR_NAMES[col]:6s} {rr}   spread {d['rank_spread']}")
