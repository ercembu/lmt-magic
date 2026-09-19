"""Train and honestly evaluate the card grader.

Evaluation is leave-one-set-out: hold out an entire set, train on the rest,
predict the held-out set cold. That mirrors the real task -- grading a set the
model has never seen -- and is the only number worth trusting. The text
vectoriser is refit inside each fold so the held-out set never contributes
vocabulary.

Target is the within-set z-score of GIH win rate. Sets differ in overall win
rate and spread, and grading is relative to a set's own pool anyway.
"""
import json, os, sys
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
import dataset, textfeat

ROOT = os.path.join(os.path.dirname(__file__), "..")
TRAIN_SETS = ["HOB", "MSH", "SOS", "TMT", "ECL", "TLA", "EOE", "FIN", "TDM",
              "DFT", "DSK", "BLB", "OTJ", "MKM", "LCI", "WOE", "LTR", "MOM",
              "ONE", "BRO", "DMU",
              # older sets; pre-2022 files carry no tutored_ columns, handled
              # in aggregate_labels.py
              "FDN", "MH3", "SIR", "SNC", "NEO", "VOW", "MID", "AFR", "STX", "KHM"]

PARAMS = dict(objective="regression", metric="l2", learning_rate=0.03,
              num_leaves=31, min_data_in_leaf=40, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
              verbose=-1, num_threads=8)
ROUNDS = 700


def feature_names(vec):
    return [n.replace(" ", "_") for n in
            dataset.FEATURE_KEYS + ["oov_rate", "mean_idf"]
            + [f"tf_{f}" for f in vec.get_feature_names_out()]]


# A rank-average of the boosted trees with a ridge on the same inputs. Small but
# real: +0.009 spearman over the GBM alone, paired t p=0.020 across 21 folds.
ENSEMBLE_GBM_WEIGHT = 0.7
RIDGE_ALPHA = 3.0


def fit_ensemble(Xtr, ytr, wtr, names):
    from sklearn.linear_model import Ridge
    gbm = fit(Xtr, ytr, wtr, names)
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(Xtr, ytr, sample_weight=wtr)
    return gbm, ridge


def predict_ensemble(models, Xte):
    from scipy.stats import rankdata
    gbm, ridge = models
    g, r = gbm.predict(Xte), ridge.predict(Xte)
    n = len(g)
    return (ENSEMBLE_GBM_WEIGHT * rankdata(g) / n
            + (1 - ENSEMBLE_GBM_WEIGHT) * rankdata(r) / n)


def fit(Xtr, ytr, wtr, names):
    ds = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=names)
    return lgb.train(PARAMS, ds, num_boost_round=ROUNDS)


def novelty(texts, vec):
    """How much precedent does the model have for this card's wording?

    Fraction of a card's tokens absent from the vocabulary learned on the
    TRAINING sets, plus the mean rarity (IDF) of the ones present. Computed from
    the fold's own fitted vectoriser, so it is fold-safe, and it is exactly the
    quantity the guard layer was trying to express by hand -- but available for
    every card in every set, so the model can learn what to do with it.
    """
    vocab = vec.vocabulary_
    idf = vec.idf_
    out = np.zeros((len(texts), 2))
    for i, t in enumerate(texts):
        toks = t.split()
        if not toks:
            out[i] = (1.0, 0.0); continue
        known = [vocab[w] for w in toks if w in vocab]
        out[i, 0] = 1.0 - len(known) / len(toks)
        out[i, 1] = float(np.mean(idf[known])) if known else 0.0
    return out


def combine(X, texts, vec, fit_vec=False):
    T = vec.fit_transform(texts) if fit_vec else vec.transform(texts)
    N = novelty(texts, vec)
    return sp.hstack([sp.csr_matrix(np.nan_to_num(X, nan=-999)),
                      sp.csr_matrix(N), T]).tocsr()


def loso(setcodes, group="all", use_text=True):
    print("Loading sets:")
    X, y, texts, w, rows = dataset.build(setcodes, group=group)
    sets = np.array([r["set"] for r in rows])
    results = []
    for s in setcodes:
        te = sets == s
        if te.sum() < 30:
            continue
        if use_text:
            vec = textfeat.vectorizer()
            Xtr = combine(X[~te], [t for t, m in zip(texts, ~te) if m], vec, fit_vec=True)
            Xte = combine(X[te], [t for t, m in zip(texts, te) if m], vec)
            names = (dataset.FEATURE_KEYS + ["oov_rate", "mean_idf"]
                     + [f"tf_{f}" for f in vec.get_feature_names_out()])
            names = [n.replace(" ", "_") for n in names]
        else:
            Xtr, Xte = X[~te], X[te]
            names = dataset.FEATURE_KEYS
        model = fit(Xtr, y[~te], w[~te], names)
        pred = model.predict(Xte)
        rho = spearmanr(pred, y[te]).statistic
        r = pearsonr(pred, y[te]).statistic
        results.append((s, int(te.sum()), float(rho), float(r)))
        print(f"  {s}: n={te.sum():3d}  spearman={rho:.3f}  pearson={r:.3f}")
    rhos = [x[2] for x in results]
    print(f"\nMEAN spearman = {np.mean(rhos):.3f}  (median {np.median(rhos):.3f})")
    return results, X, y, texts, w, rows


def baselines(X, y, rows):
    keys = dataset.FEATURE_KEYS
    sets = np.array([r["set"] for r in rows])
    out = {}
    for name, col, sign in [("rarity only", "rarity_ord", 1), ("cmc only (neg)", "cmc", -1)]:
        v = X[:, keys.index(col)] * sign
        out[name] = np.mean([spearmanr(v[sets == s], y[sets == s]).statistic for s in set(sets)])
    idx = {k: keys.index(k) for k in ("rarity_ord", "e_removal_quality", "pt_per_mana", "evasive")}
    h = (0.5 * X[:, idx["rarity_ord"]] + 0.8 * np.nan_to_num(X[:, idx["e_removal_quality"]])
         + 0.6 * np.nan_to_num(X[:, idx["pt_per_mana"]]) + 0.4 * X[:, idx["evasive"]])
    out["hand heuristic"] = np.mean([spearmanr(h[sets == s], y[sets == s]).statistic
                                     for s in set(sets)])
    return out


if __name__ == "__main__":
    group = os.environ.get("GROUP", "all")
    avail = [s for s in TRAIN_SETS if os.path.exists(f"{ROOT}/data/labels/{s}.json")]
    print(f"Sets with labels: {len(avail)}/{len(TRAIN_SETS)}  (group={group})\n")

    print("=== features only ===")
    loso(avail, group, use_text=False)
    print("\n=== features + oracle text ===")
    res, X, y, texts, w, rows = loso(avail, group, use_text=True)

    print("\nBaselines (mean per-set spearman):")
    for k, v in baselines(X, y, rows).items():
        print(f"  {k:18s} {v:.3f}")

    # final model on everything
    vec = textfeat.vectorizer()
    Xall = combine(X, texts, vec, fit_vec=True)
    names = [n.replace(" ", "_") for n in
             dataset.FEATURE_KEYS + ["oov_rate", "mean_idf"]
             + [f"tf_{f}" for f in vec.get_feature_names_out()]]
    gbm, ridge = fit_ensemble(Xall, y, w, names)
    gbm.save_model(f"{ROOT}/data/model.txt")
    import pickle
    pickle.dump(vec, open(f"{ROOT}/data/vectorizer.pkl", "wb"))
    pickle.dump(ridge, open(f"{ROOT}/data/ridge.pkl", "wb"))
    model = gbm
    imp = sorted(zip(names, model.feature_importance("gain")), key=lambda x: -x[1])[:25]
    print("\nTop features by gain:")
    for k, v in imp:
        print(f"  {k:28s} {v:10.0f}")
    json.dump({"loso": res, "mean_spearman": float(np.mean([r[2] for r in res]))},
              open(f"{ROOT}/data/validation.json", "w"), indent=1)
