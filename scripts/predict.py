"""Grade an unreleased set.

Grades are a curve within the set, the same convention limited-grades uses: a
card's grade says how it ranks against the rest of its own set, not against all
of Magic. Two consequences worth being explicit about:

  * Every set gets roughly the same number of A's. A weak set's A+ is not a
    strong set's A+.
  * Model predictions are regression-shrunk (lower variance than the truth), so
    thresholding raw predicted z-scores would pile everything into C. We rank
    the predictions and map rank -> z through the inverse normal CDF, which
    restores a realistic spread without inventing confidence we don't have.
"""
import json, os, sys, pickle
import numpy as np
import scipy.sparse as sp
from scipy.stats import norm
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
import dataset, features, textfeat, guards, train as trainmod

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Exactly the scale limited-grades uses (src/lib/CardGrader.ts), so grades here are
# directly comparable to theirs: fit a normal to the set's win rates, take each card's
# percentile under it, and cut at these thresholds.
GRADE_THRESHOLDS = [("A+", 99), ("A", 95), ("A-", 90), ("B+", 85), ("B", 76),
                    ("B-", 68), ("C+", 57), ("C", 45), ("C-", 36), ("D+", 27),
                    ("D", 17), ("D-", 5), ("F", 0)]


def grade_from_percentile(pct):
    for g, t in GRADE_THRESHOLDS:
        if pct >= t:
            return g
    return "F"


def to_grade(z):
    """z -> grade via the normal CDF, matching limited-grades' percentile cuts."""
    return grade_from_percentile(norm.cdf(z) * 100)


def empirical_grade_curve(min_sets=5, exclude=()):
    """The grade distribution released sets actually have.

    Forcing predictions onto a normal is wrong in a specific way: real win rates
    are fat-tailed (kurtosis +0.2 to +2.8 across the 21 sets), so limited-grades'
    own method hands out noticeably more A+ than a normal would. Rather than
    inherit that error, we measure the average shape across released sets and map
    predicted ranks onto it. Monotonic, so it cannot change the ordering -- only
    where the grade boundaries fall.
    """
    import dataset as _ds
    from scipy.stats import norm as _norm
    lab = os.path.join(ROOT, "data", "labels")
    # label files for non-draft formats are named "<SET>.<Format>.json"; the
    # grade curve is built from Premier Draft only
    codes = [f[:-5] for f in os.listdir(lab)
             if f.endswith(".json") and "." not in f[:-5]] if os.path.isdir(lab) else []
    counts = {g: 0 for g, _ in GRADE_THRESHOLDS}
    n_sets = 0
    for c in codes:
        if c in exclude:
            continue
        rows = [r for r in _ds.load_set(c)
                if r["gih_wr"] is not None and r["gih_n"] >= _ds.MIN_GAMES]
        if len(rows) < 100:
            continue
        wr = np.array([r["gih_wr"] for r in rows])
        for p in _norm.cdf((wr - wr.mean()) / wr.std()) * 100:
            counts[grade_from_percentile(p)] += 1
        n_sets += 1
    if n_sets < min_sets:
        return None
    total = sum(counts.values())
    return [(g, counts[g] / total) for g, _ in GRADE_THRESHOLDS]


# When a model is uncertain, the grade that minimises expected error is pulled
# toward the middle -- predicting A+ on a card you are unsure about costs more
# than predicting B. Measured over 31 sets, shrinking predicted z by 0.85 before
# cutting grades improved exact agreement 18.0%->18.6%, within-one 47.8%->48.7%
# and mean grade error 1.99->1.89. Below ~0.7 it degenerates: A+ grades vanish
# and the output stops looking like a real set.
# Default 1.0: shrinking improves the grade metrics slightly but collapses the top
# of the scale (FRA went from 5 A+ to 1), and it does not change the card ORDERING
# at all -- only the letters printed on it. Since deck quality depends entirely on
# ordering (pool_sim.py), the trade buys a metric and costs the tool its main job.
# Set to 0.85 if you want the grade numbers optimised instead.
SHRINK = 1.0


def assign_grades(scores, curve=None, shrink=SHRINK):
    """Grade by rank, in the proportions released sets actually show."""
    curve = curve or empirical_grade_curve()
    if shrink and shrink != 1.0:
        z = rank_to_z(scores) * shrink
        return [grade_from_percentile(p) for p in norm.cdf(z) * 100]
    n = len(scores)
    order = np.argsort(-np.asarray(scores))     # best first
    out = [None] * n
    i = 0
    for k, (g, frac) in enumerate(curve):
        take = n - i if k == len(curve) - 1 else int(round(frac * n))
        for j in order[i:i + take]:
            out[j] = g
        i += take
        if i >= n:
            break
    for j in range(n):
        if out[j] is None:
            out[j] = curve[-1][0]
    return out


def rank_to_z(pred):
    """Map predicted ordering onto a normal curve."""
    n = len(pred)
    order = np.argsort(np.argsort(pred))          # 0..n-1 ranks
    q = (order + 0.5) / n
    return norm.ppf(q)


def grade_set(setcode, model_path=None, exclude_unplayable=True):
    cards = json.load(open(f"{ROOT}/data/cards/{setcode}.json"))
    if exclude_unplayable:
        cards = [c for c in cards if "Basic Land" not in c.get("type_line", "")]

    model = lgb.Booster(model_file=model_path or f"{ROOT}/data/model.txt")
    vec = pickle.load(open(f"{ROOT}/data/vectorizer.pkl", "rb"))
    ridge = pickle.load(open(f"{ROOT}/data/ridge.pkl", "rb"))
    calib = None
    cp = f"{ROOT}/data/calibration.json"
    if os.path.exists(cp):
        calib = json.load(open(cp))

    fc = dataset.set_features(setcode)
    X = np.array([[fc[c["name"]].get(k, np.nan) for k in dataset.FEATURE_KEYS]
                  for c in cards])
    texts = [textfeat.normalize(c) for c in cards]
    Xf = trainmod.combine(X, texts, vec)
    nov = trainmod.novelty(texts, vec)[:, 1]
    raw = trainmod.predict_ensemble((model, ridge), Xf)
    base_z = rank_to_z(raw)                    # kept for the score display
    curve = empirical_grade_curve()
    letters = assign_grades(raw, curve) if curve else [to_grade(z) for z in base_z]

    def band_for(v):
        """Which calibration band this card's wording falls in, and what that
        historically meant for accuracy."""
        if not calib:
            return None
        for i, b in enumerate(calib["bins"]):
            lo = -np.inf if b["lo"] is None else b["lo"]
            hi = np.inf if b["hi"] is None else b["hi"]
            if lo <= v < hi:
                return {"band": i + 1, "of": len(calib["bins"]),
                        "mean_grade_error": b["mean_grade_error"],
                        "within_1": b["within_1"], "within_2": b["within_2"]}
        return None

    out = []
    for c, rz, rp, letter, nv in zip(cards, base_z, raw, letters, nov):
        adj_z, adjustments, flags, conf = guards.apply(c, setcode, rz)
        fits = guards.archetype_fit(c, setcode)
        fr = c["card_faces"][0] if "card_faces" in c else c
        out.append({
            "name": c["name"],
            "set": setcode,
            "rarity": c["rarity"],
            "mana_cost": fr.get("mana_cost", ""),
            "cmc": c.get("cmc", 0),
            "colors": c.get("colors", fr.get("colors", [])) or [],
            "color_identity": c.get("color_identity", []),
            "type_line": c.get("type_line", ""),
            "oracle_text": (c.get("oracle_text")
                            or "\n//\n".join(f.get("oracle_text", "")
                                             for f in c.get("card_faces", []))),
            "power": fr.get("power"), "toughness": fr.get("toughness"),
            "image": (c.get("image_uris") or fr.get("image_uris") or {}).get("normal"),
            "scryfall_uri": c.get("scryfall_uri"),
            "raw_score": float(rp),
            "base_z": float(rz),
            "z": float(adj_z),
            "grade": letter,
            "base_grade": letter,
            "confidence": conf,
            "trust": band_for(nv),
            "novelty": float(nv),
            "adjustments": adjustments,
            "flags": flags,
            "archetypes": fits,
        })
    out.sort(key=lambda r: -r["z"])
    return out


if __name__ == "__main__":
    setcode = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    res = grade_set(setcode)
    json.dump(res, open(f"{ROOT}/data/grades_{setcode}.json", "w"), indent=1)
    import collections
    print(f"{setcode}: graded {len(res)} cards")
    print("distribution:", dict(collections.Counter(r["grade"] for r in res)))
    print("confidence: ", dict(collections.Counter(r["confidence"] for r in res)))
    print("\nTop 20:")
    for r in res[:20]:
        print(f"  {r['grade']:2s} {r['name'][:38]:38s} {r['mana_cost']:11s} "
              f"{r['rarity'][:1].upper()} {r['confidence']}")
    print("\nBottom 10:")
    for r in res[-10:]:
        print(f"  {r['grade']:2s} {r['name'][:38]:38s} {r['mana_cost']:11s} {r['rarity'][:1].upper()}")
