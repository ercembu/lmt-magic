"""How did the pre-release grades actually do, now that the set has been played?

Everything else in this project is validated by holding out a set the model was
trained on. This is the real thing: FRA was graded before anyone had played a
game, the file was frozen, and 17lands has since recorded ~300k draft games and
~60k sealed ones. Nothing here is cross-validation -- it is the prediction
against the result.

Scored the same way the main page is: limited-grades' own algorithm (fit a
normal to the set's GIH win rates, cut at fixed percentiles), so "actual grade"
means the grade limited-grades shows for that card today.
"""
import json, os, sys
from collections import Counter
import numpy as np
from scipy.stats import spearmanr, norm

sys.path.insert(0, os.path.dirname(__file__))
import predict

ROOT = os.path.join(os.path.dirname(__file__), "..")
MIN_GAMES = 400
GI = {g: i for i, (g, _) in enumerate(predict.GRADE_THRESHOLDS)}


def actual(fmt):
    """Real grades and win rates for every card with enough games."""
    suffix = "" if fmt == "PremierDraft" else f".{fmt}"
    labels = json.load(open(f"{ROOT}/data/labels/FRA{suffix}.json"))["cards"]
    out = {}
    for name, rec in labels.items():
        g = rec["all"]
        if g["gih_wr"] and g["gih_n"] >= MIN_GAMES:
            out[name] = {"wr": g["gih_wr"], "n": g["gih_n"],
                         "maindeck_rate": g.get("maindeck_rate")}
    wr = np.array([v["wr"] for v in out.values()])
    mu, sd = wr.mean(), wr.std()
    for v in out.values():
        v["pct"] = float(norm.cdf((v["wr"] - mu) / sd) * 100)
        v["grade"] = predict.grade_from_percentile(v["pct"])
    return out


def predictors():
    """Everything that made a call before the set was played."""
    grades = json.load(open(f"{ROOT}/data/grades_FRA.prerelease.json"))
    review = json.load(open(f"{ROOT}/data/reviewdata_FRA.json"))["cards"]
    ds = {r["name"]: r["rating"] for r in review if r.get("rating") is not None}
    az = {r["name"]: r["second"] for r in review if r.get("second") is not None}
    cons = {r["name"]: r["consensus"] for r in review if r.get("consensus") is not None}
    blind = {r["name"]: r["blind_z"] for r in review if r.get("blind_z") is not None}
    return {
        "our model": {c["name"]: c["z"] for c in grades},
        "  the same model, reviews withheld": blind,
        "Draftsim": ds,
        "MTG Arena Zone": az,
        "the two-reviewer consensus": cons,
    }, {c["name"]: c for c in grades}


def score(pred, truth, key="wr"):
    names = [n for n in pred if n in truth]
    if len(names) < 30:
        return None
    rho = spearmanr([pred[n] for n in names],
                    [truth[n][key] for n in names]).statistic
    return {"n": len(names), "spearman": float(rho)}


def main():
    byname = predictors()[1]
    preds = predictors()[0]
    for fmt, label in (("PremierDraft", "Premier Draft"), ("Sealed", "Sealed")):
        truth = actual(fmt)
        games = sum(v["n"] for v in truth.values())
        print(f"\n=== FRA, {label} — {len(truth)} cards over {games:,} card-games ===")
        print(f"{'predictor':36s} {'cards':>6s} {'spearman':>9s}")
        rows = []
        for name, p in preds.items():
            s = score(p, truth)
            if s:
                rows.append((name, s))
                print(f"{name:36s} {s['n']:6d} {s['spearman']:9.4f}")

        # grade-step accuracy, the number that matters when you are reading the page
        pairs = [(byname[n]["grade"], truth[n]["grade"])
                 for n in byname if n in truth]
        d = np.array([abs(GI[a] - GI[b]) for a, b in pairs])
        print(f"\ngrade distance on {len(d)} cards: exact {np.mean(d==0):.0%}, "
              f"within 1 {np.mean(d<=1):.0%}, within 2 {np.mean(d<=2):.0%}, "
              f"mean {d.mean():.2f} steps")

        # did it find the bombs?
        top = sorted(truth, key=lambda n: -truth[n]["wr"])
        ours = sorted((n for n in byname if n in truth), key=lambda n: -byname[n]["z"])
        for k in (10, 20, 30):
            hit = len(set(ours[:k]) & set(top[:k]))
            print(f"  top {k:2d}: {hit}/{k} overlap with the real top {k}"
                  f"   (chance {k*k/len(truth):.1f})")

        if fmt == "PremierDraft":
            worst = sorted(((abs(GI[byname[n]['grade']] - GI[truth[n]['grade']]), n)
                            for n in byname if n in truth), reverse=True)[:10]
            print("\n  biggest misses:")
            for dist, n in worst:
                print(f"    {n[:34]:36s} we said {byname[n]['grade']:>2s}, "
                      f"actually {truth[n]['grade']:>2s}  ({dist} steps, "
                      f"{truth[n]['wr']*100:.1f}% GIH)")


if __name__ == "__main__":
    main()
