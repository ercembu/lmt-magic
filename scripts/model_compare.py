"""Is L2 regression the right objective here?

The pipeline discards predicted magnitudes and keeps only the ordering, so an
objective that spends capacity fitting exact win rates may be solving a harder
problem than we need. This compares objectives under identical leave-one-set-out
folds, scored both on ranking (Spearman) and on the thing we actually ship
(agreement with real limited-grades grades).
"""
import os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr, rankdata
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
import dataset, textfeat, train, predict

ROOT = os.path.join(os.path.dirname(__file__), "..")
GI = {g: i for i, (g, _) in enumerate(predict.GRADE_THRESHOLDS)}
BASE = dict(verbose=-1, num_threads=8, learning_rate=0.03, num_leaves=31,
            min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=2.0)
ROUNDS = 700
NDCG_LEVELS = 32


def fit_predict(cfg, Xtr, ytr, wtr, gtr, Xte, names):
    kind = cfg["kind"]
    if kind == "linear":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=cfg.get("alpha", 3.0))
        m.fit(Xtr, ytr, sample_weight=wtr)
        return m.predict(Xte)
    params = {**BASE, **cfg.get("params", {})}
    if kind == "rank":
        # relevance must be small non-negative ints; use within-set rank buckets
        lab = np.zeros(len(ytr), dtype=int)
        off = 0
        for n in gtr:
            seg = ytr[off:off + n]
            lab[off:off + n] = np.floor(
                (rankdata(seg) - 1) / max(n - 1, 1) * (NDCG_LEVELS - 1)).astype(int)
            off += n
        ds = lgb.Dataset(Xtr, label=lab, weight=wtr, group=gtr, feature_name=names)
    else:
        ds = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=names)
    return lgb.train(params, ds, num_boost_round=ROUNDS).predict(Xte)


def evaluate(cfg, X, y, texts, w, rows, sets):
    rhos, errs, exacts = [], [], []
    for s in sorted(set(sets)):
        te = sets == s
        if te.sum() < 30:
            continue
        tr = ~te
        vec = textfeat.vectorizer()
        Xtr = train.combine(X[tr], [t for t, m in zip(texts, tr) if m], vec, fit_vec=True)
        Xte = train.combine(X[te], [t for t, m in zip(texts, te) if m], vec)
        names = [n.replace(" ", "_") for n in
                 dataset.FEATURE_KEYS + [f"tf_{f}" for f in vec.get_feature_names_out()]]
        # groups must follow row order within the training block
        tr_sets = sets[tr]
        gtr = [int((tr_sets == q).sum()) for q in sorted(set(tr_sets))]
        idx = np.argsort([sorted(set(tr_sets)).index(q) for q in tr_sets], kind="stable")
        pred = fit_predict(cfg, Xtr[idx], y[tr][idx], w[tr][idx], gtr, Xte, names)

        wr = np.array([r["gih_wr"] for r, m in zip(rows, te) if m])
        curve = predict.empirical_grade_curve(exclude=(s,))
        pg = predict.assign_grades(pred, curve)
        ag = predict.assign_grades(wr, curve)
        e = np.array([abs(GI[a] - GI[b]) for a, b in zip(pg, ag)])
        rhos.append(spearmanr(pred, wr).statistic)
        errs.append(e.mean()); exacts.append(np.mean([a == b for a, b in zip(pg, ag)]))
    return np.mean(rhos), np.mean(errs), np.mean(exacts)


CONFIGS = [
    ("L2 regression (current)", {"kind": "reg", "params": {"objective": "regression"}}),
    ("Huber regression",        {"kind": "reg", "params": {"objective": "huber", "alpha": 1.0}}),
    ("MAE regression",          {"kind": "reg", "params": {"objective": "regression_l1"}}),
    ("LambdaRank (NDCG)",       {"kind": "rank", "params": {"objective": "lambdarank",
                                                            "ndcg_eval_at": [20],
                                                            "label_gain": list(range(NDCG_LEVELS))}}),
    ("rank_xendcg",             {"kind": "rank", "params": {"objective": "rank_xendcg",
                                                            "label_gain": list(range(NDCG_LEVELS))}}),
    ("Ridge on same inputs",    {"kind": "linear", "alpha": 3.0}),
]

if __name__ == "__main__":
    sets_avail = [s for s in train.TRAIN_SETS
                  if os.path.exists(f"{ROOT}/data/labels/{s}.json")]
    X, y, texts, w, rows = dataset.build(sets_avail, verbose=False)
    sets = np.array([r["set"] for r in rows])
    print(f"{len(rows)} cards, {len(sets_avail)} sets, leave-one-set-out\n")
    print(f"{'objective':26s} {'spearman':>9s} {'grade err':>10s} {'exact':>7s} {'secs':>6s}")
    print("-" * 62)
    best = None
    for name, cfg in CONFIGS:
        t0 = time.time()
        try:
            rho, err, ex = evaluate(cfg, X, y, texts, w, rows, sets)
        except Exception as e:
            print(f"{name:26s}  FAILED: {type(e).__name__}: {str(e)[:40]}")
            continue
        print(f"{name:26s} {rho:9.3f} {err:10.3f} {ex:6.1%} {time.time()-t0:6.0f}")
        if best is None or rho > best[1]:
            best = (name, rho)
    print("-" * 62)
    print(f"best by spearman: {best[0]} ({best[1]:.3f})")
