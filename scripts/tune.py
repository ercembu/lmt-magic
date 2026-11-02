"""Leave-one-set-out harness for trying a change and keeping it only if it earns it.

The model's hyperparameters were set by hand at the start and never searched.
Now that FRA has real labels and the bonus sheets are in, there are 32 sets and
~9k cards to search against -- and a post-mortem that says the blind model
(0.528 on FRA) is the weak half, since that is what runs until a set review is
published.

Every variant is scored against the same baseline on the same folds and keeps
only if a paired test says so. Pass --blind to drop the expert rating, which is
the state the model is in on prerelease week.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.stats import spearmanr, wilcoxon, ttest_rel

sys.path.insert(0, os.path.dirname(__file__))
import dataset, textfeat, train

ROOT = os.path.join(os.path.dirname(__file__), "..")


def folds(blind=False):
    sets = [s for s in train.TRAIN_SETS if os.path.exists(f"{ROOT}/data/labels/{s}.json")]
    X, y, texts, w, rows = dataset.build(sets, verbose=False)
    if blind:
        X = X.copy()
        X[:, dataset.FEATURE_KEYS.index("expert_rating")] = np.nan
    arr = np.array([r["set"] for r in rows])
    return [(s, arr == s) for s in sets if (arr == s).sum() >= 30], X, y, texts, w, arr


def run(X, y, texts, w, arr, flds, params=None, rounds=None, gbm_weight=None,
        ridge_alpha=None, vec_factory=None):
    """One LOSO sweep under a configuration, returning per-set spearman."""
    old = (train.PARAMS.copy(), train.ROUNDS, train.ENSEMBLE_GBM_WEIGHT, train.RIDGE_ALPHA)
    if params:       train.PARAMS.update(params)
    if rounds:       train.ROUNDS = rounds
    if gbm_weight is not None:  train.ENSEMBLE_GBM_WEIGHT = gbm_weight
    if ridge_alpha is not None: train.RIDGE_ALPHA = ridge_alpha
    out = {}
    try:
        for s, te in flds:
            tr = ~te
            vec = (vec_factory or textfeat.vectorizer)()
            Xtr = train.combine(X[tr], [t for t, m in zip(texts, tr) if m], vec, fit_vec=True)
            Xte = train.combine(X[te], [t for t, m in zip(texts, te) if m], vec)
            p = train.predict_ensemble(
                train.fit_ensemble(Xtr, y[tr], w[tr], train.feature_names(vec)), Xte)
            out[s] = float(spearmanr(p, y[te]).statistic)
    finally:
        train.PARAMS.clear(); train.PARAMS.update(old[0])
        train.ROUNDS, train.ENSEMBLE_GBM_WEIGHT, train.RIDGE_ALPHA = old[1], old[2], old[3]
    return out


def compare(base, variant, name):
    ks = sorted(set(base) & set(variant))
    a = np.array([base[k] for k in ks]); b = np.array([variant[k] for k in ks])
    d = b - a
    t = ttest_rel(b, a).pvalue
    wp = wilcoxon(b, a).pvalue if np.any(d) else 1.0
    print(f"  {name:38s} {b.mean():.4f}  {d.mean():+.4f}  "
          f"{int((d > 0).sum())}/{len(d)}  t={t:.3f} w={wp:.3f}"
          + ("   KEEP" if d.mean() > 0 and t < 0.05 else ""))
    return {"name": name, "mean": float(b.mean()), "delta": float(d.mean()),
            "improved": int((d > 0).sum()), "n": len(d), "p_t": float(t), "p_w": float(wp)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    flds, X, y, texts, w, arr = folds(blind=a.blind)
    print(f"{len(flds)} folds, {len(y)} cards, expert feature "
          f"{'WITHHELD (prerelease state)' if a.blind else 'included'}\n")

    t0 = time.time()
    base = run(X, y, texts, w, arr, flds)
    print(f"baseline {np.mean(list(base.values())):.4f}   ({time.time()-t0:.0f}s a sweep)\n")
    print(f"  {'variant':38s} {'mean':>6s}  {'delta':>7s}  {'won':>5s}  tests")

    # Capacity: the trees were given 31 leaves and 700 rounds on day one, when
    # there were 21 sets and no bonus-sheet rows. There are 32 and ~9k cards now.
    trials = [
        ("deeper trees (63 leaves)",        dict(params={"num_leaves": 63})),
        ("shallower trees (15 leaves)",     dict(params={"num_leaves": 15})),
        ("smaller leaves (min_data 20)",    dict(params={"min_data_in_leaf": 20})),
        ("larger leaves (min_data 80)",     dict(params={"min_data_in_leaf": 80})),
        ("slower + longer (0.015, 1500)",   dict(params={"learning_rate": 0.015}, rounds=1500)),
        ("more rounds (1200)",              dict(rounds=1200)),
        ("stronger l2 (8)",                 dict(params={"lambda_l2": 8.0})),
        ("thinner features (0.4)",          dict(params={"feature_fraction": 0.4})),
        ("ridge-heavier ensemble (0.5)",    dict(gbm_weight=0.5)),
        ("gbm-heavier ensemble (0.85)",     dict(gbm_weight=0.85)),
        ("weaker ridge (alpha 10)",         dict(ridge_alpha=10.0)),
    ]
    res = []
    for name, kw in trials:
        res.append(compare(base, run(X, y, texts, w, arr, flds, **kw), name))
    if a.out:
        json.dump({"blind": a.blind, "baseline": base, "trials": res},
                  open(a.out, "w"), indent=1)
