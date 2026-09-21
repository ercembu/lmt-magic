"""Compare predicted grades to limited-grades' ACTUAL grades, on every released set.

limited-grades computes a grade by fitting a normal to the set's GIH win rates and
cutting each card's percentile at fixed thresholds (src/lib/CardGrader.ts). We use
that exact algorithm, so 'actual grade' here is the grade limited-grades shows.

Every set is graded by a model trained only on the other 20, so nothing here is a
set the model has seen.
"""
import json, os, sys, collections
import numpy as np
from scipy.stats import spearmanr, norm

sys.path.insert(0, os.path.dirname(__file__))
import dataset, textfeat, train, predict

ROOT = os.path.join(os.path.dirname(__file__), "..")
GI = {g: i for i, (g, _) in enumerate(predict.GRADE_THRESHOLDS)}


def actual_grades(wr):
    """limited-grades' algorithm, applied to real win rates."""
    mu, sd = np.mean(wr), np.std(wr)
    pct = norm.cdf((np.asarray(wr) - mu) / sd) * 100
    return [predict.grade_from_percentile(p) for p in pct], pct


def run():
    sets = [s for s in train.TRAIN_SETS if os.path.exists(f"{ROOT}/data/labels/{s}.json")]
    X, y, texts, w, rows = dataset.build(sets, verbose=False)
    arr = np.array([r["set"] for r in rows])

    per_set, all_err, all_pred, all_act = [], [], [], []
    for s in sets:
        te = arr == s
        if te.sum() < 30:
            continue
        vec = textfeat.vectorizer()
        Xtr = train.combine(X[~te], [t for t, m in zip(texts, ~te) if m], vec, fit_vec=True)
        Xte = train.combine(X[te], [t for t, m in zip(texts, te) if m], vec)
        names = train.feature_names(vec)
        pred = train.predict_ensemble(train.fit_ensemble(Xtr, y[~te], w[~te], names), Xte)

        sub = [r for r, m in zip(rows, te) if m]
        # never-played cards are training aids carrying a placeholder win rate;
        # scoring predictions against that placeholder measures nothing
        real = [i for i, r in enumerate(sub) if not r.get("unplayed")]
        pred = pred[real]
        wr = np.array([sub[i]["gih_wr"] for i in real])
        act, _ = actual_grades(wr)
        # predicted grades: rank predictions onto the grade distribution that the
        # OTHER sets show -- the held-out set contributes nothing to the curve
        curve = predict.empirical_grade_curve(exclude=(s,))
        pg = predict.assign_grades(pred, curve)

        err = np.array([abs(GI[a] - GI[b]) for a, b in zip(pg, act)])
        exact = np.mean([a == b for a, b in zip(pg, act)])
        rho = spearmanr(pred, wr).statistic
        per_set.append((s, len(real), rho, exact, err.mean(),
                        float((err <= 1).mean()), float((err <= 2).mean())))
        all_err.append(err); all_pred += pg; all_act += act

    all_err = np.concatenate(all_err)
    print(f"{'set':5s} {'n':>4s} {'spear':>6s} {'exact':>6s} {'mean err':>9s} "
          f"{'within 1':>9s} {'within 2':>9s}")
    print("-" * 54)
    for s, n, rho, ex, me, w1, w2 in per_set:
        print(f"{s:5s} {n:4d} {rho:6.3f} {ex:5.0%} {me:9.2f} {w1:8.0%} {w2:8.0%}")
    print("-" * 54)
    print(f"{'ALL':5s} {len(all_err):4d} "
          f"{np.mean([p[2] for p in per_set]):6.3f} "
          f"{np.mean([a == b for a, b in zip(all_pred, all_act)]):5.0%} "
          f"{all_err.mean():9.2f} {(all_err <= 1).mean():8.0%} {(all_err <= 2).mean():8.0%}")

    print("\nHow often does the model call a bomb a bomb, or a dud a dud?")
    for label, gs in [("A-tier (A+/A/A-)", {"A+", "A", "A-"}),
                      ("B-tier", {"B+", "B", "B-"}),
                      ("C-tier", {"C+", "C", "C-"}),
                      ("D/F-tier", {"D+", "D", "D-", "F"})]:
        idx = [i for i, a in enumerate(all_act) if a in gs]
        if not idx:
            continue
        same = np.mean([all_pred[i] in gs for i in idx])
        print(f"  actually {label:18s} n={len(idx):4d}  "
              f"model put {same:.0%} in the same tier")

    print("\nBiggest systematic drift (predicted tier vs actual tier):")
    cm = collections.Counter((a[0], p[0]) for a, p in zip(all_act, all_pred))
    for (a, p), n in cm.most_common(8):
        print(f"  actual {a}-tier -> predicted {p}-tier: {n}")
    return per_set


if __name__ == "__main__":
    run()
