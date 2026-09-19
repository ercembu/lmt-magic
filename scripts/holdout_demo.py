"""Grade a released set with a model that never saw it, then show the damage.

Spearman is a summary; this prints the thing you actually want to know before
trusting a grade: when the model says A+, what happened? When it says F, was the
card really unplayable? Run it on any released set.
"""
import json, os, sys
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr, norm
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
import dataset, textfeat, train, predict

ROOT = os.path.join(os.path.dirname(__file__), "..")


def run(holdout):
    others = [s for s in train.TRAIN_SETS
              if s != holdout and os.path.exists(f"{ROOT}/data/labels/{s}.json")]
    X, y, texts, w, rows = dataset.build(others, verbose=False)
    vec = textfeat.vectorizer()
    Xtr = train.combine(X, texts, vec, fit_vec=True)
    names = train.feature_names(vec)
    model = train.fit_ensemble(Xtr, y, w, names)

    Xh, yh, th, wh, rh = dataset.build([holdout], verbose=False)
    Xhf = train.combine(Xh, th, vec)
    pred = train.predict_ensemble(model, Xhf)

    pz = predict.rank_to_z(pred)
    az = predict.rank_to_z(yh)          # actual, on the same curve
    pg = [predict.to_grade(z) for z in pz]
    ag = [predict.to_grade(z) for z in az]
    gi = {g: i for i, (g, _) in enumerate(predict.GRADES)}

    rho = spearmanr(pred, yh).statistic
    order = np.argsort(-pred)
    actual_pct = (np.argsort(np.argsort(yh)) / (len(yh) - 1))

    print(f"=== {holdout}: {len(rh)} cards, model trained on {len(others)} other sets ===")
    print(f"Spearman {rho:.3f}\n")

    print("MODEL'S TOP 15 PICKS         grade  actual   actual  GIH")
    print("                             pred/real percentile  WR")
    for i in order[:15]:
        print(f"  {rh[i]['name'][:28]:28s} {pg[i]:>3s}/{ag[i]:<3s} "
              f"{actual_pct[i]*100:8.0f}%  {rh[i]['gih_wr']:.3f}")
    print("\nMODEL'S BOTTOM 10 PICKS")
    for i in order[-10:]:
        print(f"  {rh[i]['name'][:28]:28s} {pg[i]:>3s}/{ag[i]:<3s} "
              f"{actual_pct[i]*100:8.0f}%  {rh[i]['gih_wr']:.3f}")

    top10 = order[:10]
    print(f"\nModel's top 10 landed at median actual percentile: "
          f"{np.median(actual_pct[top10])*100:.0f}%")
    bot10 = order[-10:]
    print(f"Model's bottom 10 landed at median actual percentile: "
          f"{np.median(actual_pct[bot10])*100:.0f}%")

    真 = np.argsort(-yh)[:20]
    print(f"Of the set's 20 genuinely best cards, model put "
          f"{sum(1 for i in 真 if list(order).index(i) < 40)}/20 in its top 40")
    err = np.array([abs(gi[p] - gi[a]) for p, a in zip(pg, ag)])
    print(f"Mean grade error: {err.mean():.2f} steps (e.g. B+ -> B- is 2)")
    print(f"Within 1 grade step: {(err <= 1).mean():.0%}   within 3: {(err <= 3).mean():.0%}")
    return rho


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "BLB")
