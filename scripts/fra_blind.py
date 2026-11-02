"""The one number that matters on prerelease week: blind accuracy on a real set.

FRA is the only set we predicted with nothing but card text and past win rates,
and it has since been played ~300k times. So: train on every other set, predict
FRA with the expert rating withheld, and score against what actually happened.
That is exactly the situation TRK is in on Friday morning before any review is
published.

The post-mortem measured this at 0.528, against 0.675 once the reviews were fed
in. Closing that gap is worth more than anything else available before reviews
land.
"""
import argparse, json, os, sys
import numpy as np
from scipy.stats import spearmanr, norm

sys.path.insert(0, os.path.dirname(__file__))
import dataset, textfeat, train, predict

ROOT = os.path.join(os.path.dirname(__file__), "..")
GI = {g: i for i, (g, _) in enumerate(predict.GRADE_THRESHOLDS)}


def truth(fmt="PremierDraft"):
    suffix = "" if fmt == "PremierDraft" else f".{fmt}"
    lab = json.load(open(f"{ROOT}/data/labels/FRA{suffix}.json"))["cards"]
    out = {n: r["all"] for n, r in lab.items()
           if r["all"]["gih_wr"] and r["all"]["gih_n"] >= 400}
    wr = np.array([v["gih_wr"] for v in out.values()])
    mu, sd = wr.mean(), wr.std()
    for v in out.values():
        v["grade"] = predict.grade_from_percentile(norm.cdf((v["gih_wr"] - mu) / sd) * 100)
    return out


def build(blind=True):
    sets = [s for s in train.TRAIN_SETS
            if s != "FRA" and os.path.exists(f"{ROOT}/data/labels/{s}.json")]
    X, y, texts, w, rows = dataset.build(sets, verbose=False)
    cards = [c for c in json.load(open(f"{ROOT}/data/cards/FRA.json"))
             if "Basic Land" not in c.get("type_line", "")]
    fc = dataset.set_features("FRA")
    Xh = np.array([[fc[c["name"]].get(k, np.nan) for k in dataset.FEATURE_KEYS]
                   for c in cards])
    if blind:
        Xh[:, dataset.FEATURE_KEYS.index("expert_rating")] = np.nan
    return (X, y, texts, w), (cards, Xh, [textfeat.normalize(c) for c in cards])


def evaluate(trainset, held, label="", **cfg):
    X, y, texts, w = trainset
    cards, Xh, th = held
    old = (train.PARAMS.copy(), train.ROUNDS, train.ENSEMBLE_GBM_WEIGHT, train.RIDGE_ALPHA)
    if cfg.get("params"):  train.PARAMS.update(cfg["params"])
    if cfg.get("rounds"):  train.ROUNDS = cfg["rounds"]
    if cfg.get("gbm_weight") is not None:  train.ENSEMBLE_GBM_WEIGHT = cfg["gbm_weight"]
    if cfg.get("ridge_alpha") is not None: train.RIDGE_ALPHA = cfg["ridge_alpha"]
    try:
        vec = (cfg.get("vec_factory") or textfeat.vectorizer)()
        Xtr = train.combine(X, texts, vec, fit_vec=True)
        models = train.fit_ensemble(Xtr, y, w, train.feature_names(vec))
        p = train.predict_ensemble(models, train.combine(Xh, th, vec))
    finally:
        train.PARAMS.clear(); train.PARAMS.update(old[0])
        train.ROUNDS, train.ENSEMBLE_GBM_WEIGHT, train.RIDGE_ALPHA = old[1], old[2], old[3]

    z = predict.rank_to_z(p)
    out = {}
    for fmt in ("PremierDraft", "Sealed"):
        t = truth(fmt)
        names = [c["name"] for c in cards if c["name"] in t]
        idx = [i for i, c in enumerate(cards) if c["name"] in t]
        out[fmt] = float(spearmanr([z[i] for i in idx],
                                   [t[n]["gih_wr"] for n in names]).statistic)
    t = truth()
    grades = predict.assign_grades(z, predict.empirical_grade_curve(exclude="FRA"))
    pairs = [(grades[i], t[c["name"]]["grade"])
             for i, c in enumerate(cards) if c["name"] in t]
    d = np.array([abs(GI[a] - GI[b]) for a, b in pairs])
    out["within1"] = float(np.mean(d <= 1))
    out["steps"] = float(d.mean())
    print(f"  {label:38s} draft {out['PremierDraft']:.4f}  sealed {out['Sealed']:.4f}  "
          f"within1 {out['within1']:.0%}  {out['steps']:.2f} steps", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sighted", action="store_true", help="feed it the reviews too")
    a = ap.parse_args()
    trainset, held = build(blind=not a.sighted)
    print(f"trained on {len(trainset[1])} cards from 31 sets, predicting FRA "
          f"{'WITH reviews' if a.sighted else 'BLIND'}\n")
    evaluate(trainset, held, "baseline")
