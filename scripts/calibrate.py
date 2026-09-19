"""Learn how much to trust a grade, from how unusual the card's wording is.

Nudging grades by hand failed its ablation, and feeding guard signals in as
features changed nothing (they are redundant with the text features). What does
survive testing is uncertainty: cards whose wording is rare relative to the
training corpus are measurably harder for the model.

This runs out-of-fold predictions over every released set, measures actual grade
error against novelty, and saves a calibration table. predict.py then reports an
honest error band per card instead of a hand-waved confidence label.
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr, rankdata
import lightgbm as lgb
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(__file__))
import dataset, textfeat, train, predict

ROOT = os.path.join(os.path.dirname(__file__), "..")
GI = {g: i for i, (g, _) in enumerate(predict.GRADE_THRESHOLDS)}
N_BINS = 5


def oof():
    sets = [s for s in train.TRAIN_SETS
            if os.path.exists(f"{ROOT}/data/labels/{s}.json")]
    X, y, texts, w, rows = dataset.build(sets, verbose=False)
    arr = np.array([r["set"] for r in rows])
    err = np.zeros(len(rows)); nov = np.zeros(len(rows))
    for s in sets:
        te = arr == s
        if te.sum() < 30:
            continue
        tr = ~te
        vec = textfeat.vectorizer()
        Xtr = train.combine(X[tr], [t for t, m in zip(texts, tr) if m], vec, fit_vec=True)
        te_txt = [t for t, m in zip(texts, te) if m]
        Xte = train.combine(X[te], te_txt, vec)
        names = train.feature_names(vec)
        pred = train.predict_ensemble(train.fit_ensemble(Xtr, y[tr], w[tr], names), Xte)
        wr = np.array([r["gih_wr"] for r, m in zip(rows, te) if m])
        curve = predict.empirical_grade_curve(exclude=(s,))
        pg = predict.assign_grades(pred, curve)
        ag = predict.assign_grades(wr, curve)
        err[te] = [abs(GI[a] - GI[b]) for a, b in zip(pg, ag)]
        nov[te] = train.novelty(te_txt, vec)[:, 1]      # mean term rarity (IDF)
    return err, nov


def build_table(err, nov):
    edges = np.quantile(nov, np.linspace(0, 1, N_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = []
    for i in range(N_BINS):
        m = (nov >= edges[i]) & (nov < edges[i + 1])
        if not m.any():
            continue
        bins.append({
            "lo": None if i == 0 else float(edges[i]),
            "hi": None if i == N_BINS - 1 else float(edges[i + 1]),
            "n": int(m.sum()),
            "mean_grade_error": float(err[m].mean()),
            "within_1": float((err[m] <= 1).mean()),
            "within_2": float((err[m] <= 2).mean()),
        })
    return {"metric": "mean_idf", "bins": bins,
            "spearman_novelty_vs_error": float(spearmanr(nov, err).statistic),
            "overall_mean_error": float(err.mean())}


if __name__ == "__main__":
    err, nov = oof()
    tbl = build_table(err, nov)
    json.dump(tbl, open(f"{ROOT}/data/calibration.json", "w"), indent=1)
    print(f"novelty vs error spearman: {tbl['spearman_novelty_vs_error']:+.3f}   "
          f"overall mean error {tbl['overall_mean_error']:.2f}\n")
    print(f"{'band':6s} {'n':>5s} {'mean err':>9s} {'within 1':>9s} {'within 2':>9s}")
    for i, b in enumerate(tbl["bins"]):
        print(f"  {i+1:<4d} {b['n']:5d} {b['mean_grade_error']:9.2f} "
              f"{b['within_1']:8.0%} {b['within_2']:8.0%}")
