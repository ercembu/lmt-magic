"""Data for the reviewer page: his ratings, his track record, and where we disagree.

Only one set review is scrapable per set (MTG Arena Zone blocks bots, CFB is a
JS app), so there is no consensus to average across sources. What can be
aggregated is his *record*: 25 past sets where we have both his ratings and what
actually happened.
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import dataset, textfeat, train, predict, mtgazone
from dataset import MIN_GAMES

ROOT = os.path.join(os.path.dirname(__file__), "..")
GI = {g: i for i, (g, _) in enumerate(predict.GRADE_THRESHOLDS)}


def track_record():
    """How well has this reviewer called past sets, against a blind model?"""
    sets = [s for s in train.TRAIN_SETS
            if os.path.exists(f"{ROOT}/data/labels/{s}.json")
            and os.path.exists(f"{ROOT}/data/expert/{s}.json")]
    out = []
    for s in sets:
        exp = json.load(open(f"{ROOT}/data/expert/{s}.json"))
        recs = [r for r in dataset.load_set(s)
                if r["gih_wr"] and r["gih_n"] >= 400 and r["name"] in exp]
        if len(recs) < 40:
            continue
        rho = spearmanr([exp[r["name"]] for r in recs],
                        [r["gih_wr"] for r in recs]).statistic
        out.append({"set": s, "n": len(recs), "spearman": float(rho)})
    return out


def _bucket(r):
    """Group a card for the bias table: rarity, cost band, and broad type."""
    t = r["type_line"]
    kind = ("creature" if "Creature" in t else
            "instant/sorcery" if ("Instant" in t or "Sorcery" in t) else
            "land" if "Land" in t else "other")
    cmc = r.get("cmc") or 0
    band = "0-2" if cmc <= 2 else "3-4" if cmc <= 4 else "5-6" if cmc <= 6 else "7+"
    return r["rarity"], band, kind


def reviewer_bias():
    """Where pre-release reviews sit relative to what actually happened.

    Every past card with both a published rating and a real win rate. Both sides
    are rank-normalised within their own set, so the comparison is "was this
    rated above or below where it finished, relative to its set" rather than
    anything scale-dependent. Positive means the review was too generous.

    The model's own out-of-fold predictions go through the same machine, because
    the obvious hope -- that a model seeing rarity and the rating side by side
    would correct the reviewer's bias -- turns out to be false, and the page
    should say so rather than imply otherwise.
    """
    oof = {(r["set"], r["name"]): r for r in json.load(open(f"{ROOT}/data/oof.json"))}
    rows = []
    sets = [s for s in train.TRAIN_SETS
            if os.path.exists(f"{ROOT}/data/labels/{s}.json")
            and os.path.exists(f"{ROOT}/data/expert/{s}.json")]
    for s in sets:
        exp = json.load(open(f"{ROOT}/data/expert/{s}.json"))
        recs = [r for r in dataset.load_set(s)
                if r["gih_wr"] and r["gih_n"] >= MIN_GAMES and r["name"] in exp]
        if len(recs) < 40:
            continue
        az = predict.rank_to_z([r["gih_wr"] for r in recs])
        rz = predict.rank_to_z([exp[r["name"]] for r in recs])
        mz = predict.rank_to_z([oof[(s, r["name"])]["pred"]
                                if (s, r["name"]) in oof else np.nan for r in recs])
        for r, a, e, m in zip(recs, az, rz, mz):
            c = r["card"]
            rows.append({"rarity": c["rarity"], "cmc": c.get("cmc", 0),
                         "type_line": c.get("type_line", ""),
                         "reviewer": float(e - a),
                         "model": float(m - a) if np.isfinite(m) else None})

    def table(idx, order):
        out = []
        for key in order:
            grp = [r for r in rows if _bucket(r)[idx] == key]
            if len(grp) < 30:
                continue
            mv = [r["model"] for r in grp if r["model"] is not None]
            out.append({"key": key, "n": len(grp),
                        "reviewer": float(np.mean([r["reviewer"] for r in grp])),
                        "model": float(np.mean(mv)) if mv else None})
        return out

    per_set = []
    for s in sets:
        exp = json.load(open(f"{ROOT}/data/expert/{s}.json"))
        recs = [r for r in dataset.load_set(s)
                if r["gih_wr"] and r["gih_n"] >= MIN_GAMES and r["name"] in exp]
        if len(recs) < 40:
            continue
        per_set.append(float(spearmanr([exp[r["name"]] for r in recs],
                                       [r["gih_wr"] for r in recs]).statistic))
    return {
        "cards": len(rows), "sets": len(per_set),
        "accuracy": {"mean": float(np.mean(per_set)),
                     "best": float(max(per_set)), "worst": float(min(per_set))},
        "rarity": table(0, ["mythic", "rare", "uncommon", "common"]),
        "cost": table(1, ["0-2", "3-4", "5-6", "7+"]),
        "type": table(2, ["creature", "instant/sorcery", "land", "other"]),
    }


def build(setcode="FRA"):
    exp_path = f"{ROOT}/data/expert/{setcode}.json"
    if not os.path.exists(exp_path):
        raise SystemExit(f"no review scraped for {setcode}")
    exp = json.load(open(exp_path))
    # second reviewer, published colour by colour -- whatever exists so far
    second, _ = mtgazone.load(setcode)
    grades = {r["name"]: r for r in json.load(open(f"{ROOT}/data/grades_{setcode}.json"))}
    prose = {p["name"]: p["prose"] for p in json.load(open(f"{ROOT}/data/prose.json"))
             if p["set"] == setcode}

    # our grade without his rating, so "disagreement" means something
    sets = [s for s in train.TRAIN_SETS if os.path.exists(f"{ROOT}/data/labels/{s}.json")]
    X, y, texts, w, rows = dataset.build(sets, verbose=False)
    vec = textfeat.vectorizer()
    models = train.fit_ensemble(train.combine(X, texts, vec, fit_vec=True), y, w,
                                train.feature_names(vec))
    cards = [c for c in json.load(open(f"{ROOT}/data/cards/{setcode}.json"))
             if "Basic Land" not in c.get("type_line", "")]
    fc = dataset.set_features(setcode)
    ei = dataset.FEATURE_KEYS.index("expert_rating")
    Xh = np.array([[fc[c["name"]].get(k, np.nan) for k in dataset.FEATURE_KEYS] for c in cards])
    Xh[:, ei] = np.nan
    blind = predict.rank_to_z(train.predict_ensemble(
        models, train.combine(Xh, [textfeat.normalize(c) for c in cards], vec)))
    blind_g = predict.assign_grades(blind, predict.empirical_grade_curve())
    bz = {c["name"]: (g, float(z)) for c, g, z in zip(cards, blind_g, blind)}

    rated = [c for c in cards if c["name"] in exp]
    ez = predict.rank_to_z([exp[c["name"]] for c in rated])
    out = []
    for c, z in zip(rated, ez):
        g = grades.get(c["name"], {})
        b_grade, b_z = bz.get(c["name"], ("?", 0.0))
        out.append({
            "name": c["name"], "rating": exp[c["name"]],
            "reviewer_z": float(z), "reviewer_grade": predict.grade_from_percentile(
                __import__("scipy.stats", fromlist=["norm"]).norm.cdf(z) * 100),
            "blind_grade": b_grade, "blind_z": b_z,
            "final_grade": g.get("grade"),
            "disagreement": float(z - b_z),
            "colors": c.get("colors", []), "rarity": c["rarity"],
            "mana_cost": c.get("mana_cost", ""), "type_line": c.get("type_line", ""),
            "image": (c.get("image_uris") or
                      (c.get("card_faces", [{}])[0].get("image_uris") or {})).get("normal"),
            "scryfall_uri": c.get("scryfall_uri"),
            "prose": prose.get(c["name"], "")[:600],
            # MTG Arena Zone rates 0-5 where Draftsim rates 0-10; both are
            # anchored at zero, so halving the Draftsim score compares them
            "second": second.get(c["name"]),
            # consensus on the 0-10 scale: the mean of the two where both have
            # rated, otherwise whichever one exists
            "consensus": (round((exp[c["name"]] + second[c["name"]] * 2) / 2, 2)
                          if second.get(c["name"]) is not None else exp[c["name"]]),
        })
    out.sort(key=lambda r: -r["rating"])
    # cards only the second reviewer covered still belong on the page
    only_second = [n for n in second if n not in exp]
    byname = {c["name"]: c for c in cards}
    for n in only_second:
        c = byname.get(n)
        if not c:
            continue
        g = grades.get(n, {})
        b_grade, b_z = bz.get(n, ("?", 0.0))
        out.append({"name": n, "rating": None, "reviewer_z": None,
                    "reviewer_grade": None, "blind_grade": b_grade, "blind_z": b_z,
                    "final_grade": g.get("grade"), "disagreement": 0.0,
                    "colors": c.get("colors", []), "rarity": c["rarity"],
                    "mana_cost": c.get("mana_cost", ""), "type_line": c.get("type_line", ""),
                    "image": (c.get("image_uris") or {}).get("normal"),
                    "scryfall_uri": c.get("scryfall_uri"), "prose": "",
                    "second": second[n], "consensus": second[n] * 2})
    out.sort(key=lambda r: -(r["consensus"] if r["consensus"] is not None else -1))

    both = [(r["rating"] / 2.0, r["second"]) for r in out
            if r["second"] is not None and r["rating"] is not None]
    agree = None
    if len(both) >= 15:
        agree = float(spearmanr([a for a, b in both], [b for a, b in both]).statistic)
    return {"set": setcode, "cards": out, "track_record": track_record(),
            "bias": reviewer_bias(),
            "unrated": [c["name"] for c in cards if c["name"] not in exp],
            "second_source": {
                "name": "MTG Arena Zone (j2sjosh)", "scale": "0-5",
                "rated": len(second), "overlap": len(both), "agreement": agree,
                "url": "https://mtgazone.com/reality-fracture-fra-limited-set-review-white/"}}


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    d = build(code)
    json.dump(d, open(f"{ROOT}/data/reviewdata_{code}.json", "w"))
    tr = [t["spearman"] for t in d["track_record"]]
    print(f"{code}: {len(d['cards'])} rated, {len(d['unrated'])} unrated")
    print(f"track record: {len(tr)} past sets, mean spearman {np.mean(tr):.3f} "
          f"(range {min(tr):.2f}-{max(tr):.2f})")
    ss = d["second_source"]
    print(f"second reviewer: {ss['rated']} cards, overlap {ss['overlap']}, "
          f"agreement with Draftsim {ss['agreement']:.3f}" if ss["agreement"] is not None
          else f"second reviewer: {ss['rated']} cards (too few to correlate)")
    big = sorted(d["cards"], key=lambda r: -abs(r["disagreement"]))[:5]
    print("biggest disagreements with the blind model:")
    for r in big:
        print(f"   {r['name'][:32]:34s} reviewer {r['rating']:.0f}/10 "
              f"vs our blind {r['blind_grade']:<2s}  ({r['disagreement']:+.2f}z)")
