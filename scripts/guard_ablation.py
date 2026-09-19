"""Do the guard adjustments help, or are they just confident noise?

Holds out a released set, grades it with and without guards, and scores both
against the set's real limited-grades grades. A guard layer that cannot beat
switching it off has no business shipping.
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr, norm

sys.path.insert(0, os.path.dirname(__file__))
import dataset, textfeat, train, predict, guards

ROOT = os.path.join(os.path.dirname(__file__), "..")
GI = {g: i for i, (g, _) in enumerate(predict.GRADE_THRESHOLDS)}


def run(holdout):
    if not os.path.exists(f"{ROOT}/data/setinfo/{holdout}.json"):
        raise SystemExit(f"no setinfo for {holdout}; run parse_guide.py first")

    others = [s for s in train.TRAIN_SETS
              if s != holdout and os.path.exists(f"{ROOT}/data/labels/{s}.json")]
    X, y, texts, w, rows = dataset.build(others, verbose=False)
    vec = textfeat.vectorizer()
    Xtr = train.combine(X, texts, vec, fit_vec=True)
    names = train.feature_names(vec)
    model = train.fit_ensemble(Xtr, y, w, names)

    Xh, yh, th, wh, rh = dataset.build([holdout], verbose=False)
    pred = train.predict_ensemble(model, train.combine(Xh, th, vec))
    base_z = predict.rank_to_z(pred)

    wr = np.array([r["gih_wr"] for r in rh])
    curve = predict.empirical_grade_curve(exclude=(holdout,))
    act = predict.assign_grades(wr, curve)

    # with guards
    guard_z, n_adj = [], 0
    for r, z in zip(rh, base_z):
        az, adj, flags, conf = guards.apply(r["card"], holdout, z, apply_adjustments=True)
        guard_z.append(az)
        n_adj += bool(adj)
    guard_z = np.array(guard_z)

    def score(z, label):
        g = predict.assign_grades(z, curve)
        err = np.array([abs(GI[a] - GI[b]) for a, b in zip(g, act)])
        rho = spearmanr(z, wr).statistic
        print(f"  {label:16s} spearman {rho:.3f}   exact {np.mean([a==b for a,b in zip(g,act)]):4.0%}"
              f"   mean err {err.mean():.2f}   within 1 {(err<=1).mean():4.0%}")
        return rho, err.mean()

    print(f"=== {holdout}: {len(rh)} cards, model trained on {len(others)} other sets")
    print(f"    guards touched {n_adj} cards")
    r0, e0 = score(base_z, "model only")
    r1, e1 = score(guard_z, "model + guards")
    d = e0 - e1
    print(f"\n  guards change mean grade error by {-d:+.3f} steps "
          f"({'better' if d > 0 else 'WORSE' if d < 0 else 'no change'}), "
          f"spearman {r1-r0:+.3f}")
    return r0, r1, e0, e1


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "MKM")
