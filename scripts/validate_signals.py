"""Check the derived signals, not just the card grades.

signals.py produces two rankings nothing has tested: which colours are deep, and
which colour pairs are strong. Both are aggregates of predicted grades, so they
could be better than the card grades (errors cancel when you average 15 cards) or
worse (a systematic colour bias would compound). Only one way to find out.

For every released set, predict it with a model that never saw it, build the same
rankings from the predictions and from the real win rates, and compare.
"""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
import dataset, textfeat, train, predict, signals

ROOT = os.path.join(os.path.dirname(__file__), "..")
WUBRG = "WUBRG"
PAIRS = [a + b for i, a in enumerate(WUBRG) for b in WUBRG[i + 1:]]
TOPN, DECK = 15, 23


def depth_scores(rows, z):
    """Mean z of each colour's best TOPN cards."""
    out = {}
    for col in WUBRG:
        vals = sorted((zz for r, zz in zip(rows, z)
                       if col in (r["card"].get("colors") or [])), reverse=True)
        out[col] = float(np.mean(vals[:TOPN])) if vals else -9.0
    return out


def pair_scores(rows, z):
    """Mean z of the best DECK cards castable in each pair."""
    out = {}
    for pair in PAIRS:
        vals = sorted((zz for r, zz in zip(rows, z)
                       if set(r["card"].get("color_identity") or []) <= set(pair)),
                      reverse=True)
        out[pair] = float(np.mean(vals[:DECK])) if vals else -9.0
    return out


def run():
    sets = [s for s in train.TRAIN_SETS
            if os.path.exists(f"{ROOT}/data/labels/{s}.json")]
    X, y, texts, w, rows = dataset.build(sets, verbose=False)
    arr = np.array([r["set"] for r in rows])

    col_rho, pair_rho = [], []
    spreads, errs, pspreads, perrs = [], [], [], []
    best_col_hit = worst_col_hit = best_pair_hit = top3_pair_hit = 0
    per_set = []
    for s in sets:
        te = arr == s
        if te.sum() < 30:
            continue
        tr = ~te
        vec = textfeat.vectorizer()
        Xtr = train.combine(X[tr], [t for t, m in zip(texts, tr) if m], vec, fit_vec=True)
        Xte = train.combine(X[te], [t for t, m in zip(texts, te) if m], vec)
        pred = train.predict_ensemble(
            train.fit_ensemble(Xtr, y[tr], w[tr], train.feature_names(vec)), Xte)

        sub = [r for r, m in zip(rows, te) if m]
        # exclude lands from both sides, as signals.py does
        keep = [i for i, r in enumerate(sub) if "Land" not in (r["card"].get("type_line") or "")]
        sub = [sub[i] for i in keep]
        wr = np.array([r["gih_wr"] for r in sub])
        # Both sides must be put on the SAME scale or the comparison is rigged:
        # predictions are rank-normalised (normal by construction) while raw win
        # rates are fat-tailed, and a mean-of-top-15 is sensitive to tail shape.
        # Rank-normalising both leaves only the card ordering to differ, which is
        # the thing the model is actually being asked to get right.
        pz = predict.rank_to_z(np.asarray(pred)[keep])
        az = predict.rank_to_z(wr)

        pd_, ad = depth_scores(sub, pz), depth_scores(sub, az)
        pp, ap = pair_scores(sub, pz), pair_scores(sub, az)
        rc = spearmanr([pd_[c] for c in WUBRG], [ad[c] for c in WUBRG]).statistic
        rp = spearmanr([pp[c] for c in PAIRS], [ap[c] for c in PAIRS]).statistic
        col_rho.append(rc); pair_rho.append(rp)

        pbest = max(WUBRG, key=lambda c: pd_[c]); abest = max(WUBRG, key=lambda c: ad[c])
        pworst = min(WUBRG, key=lambda c: pd_[c]); aworst = min(WUBRG, key=lambda c: ad[c])
        ppair = max(PAIRS, key=lambda c: pp[c]); apair = max(PAIRS, key=lambda c: ap[c])
        atop3 = sorted(PAIRS, key=lambda c: -ap[c])[:3]
        best_col_hit += pbest == abest
        worst_col_hit += pworst == aworst
        best_pair_hit += ppair == apair
        top3_pair_hit += ppair in atop3
        per_set.append((s, rc, rp, pbest, abest, ppair, apair))
        spreads.append(np.std([ad[c] for c in WUBRG]))
        errs.append(np.mean([abs(pd_[c] - ad[c]) for c in WUBRG]))
        pspreads.append(np.std([ap[c] for c in PAIRS]))
        perrs.append(np.mean([abs(pp[c] - ap[c]) for c in PAIRS]))

    n = len(per_set)
    print(f"{'set':5s} {'colour rho':>11s} {'pair rho':>9s}  {'best colour':>18s}  {'best pair':>14s}")
    print("-" * 66)
    for s, rc, rp, pb, ab, pp_, ap_ in per_set:
        print(f"{s:5s} {rc:11.2f} {rp:9.2f}  {pb+' / '+ab:>18s}{'  ✓' if pb==ab else '   '}"
              f"  {pp_+' / '+ap_:>12s}{' ✓' if pp_==ap_ else ''}")
    print("-" * 66)
    print(f"mean colour-depth rho   {np.mean(col_rho):+.3f}   (5 colours per set)")
    print(f"mean pair-strength rho  {np.mean(pair_rho):+.3f}   (10 pairs per set)")
    print(f"\nbest colour identified exactly   {best_col_hit}/{n}  (chance 1/5)")
    print(f"worst colour identified exactly  {worst_col_hit}/{n}  (chance 1/5)")
    print(f"best pair identified exactly     {best_pair_hit}/{n}  (chance 1/10)")
    print(f"best pair inside the real top 3  {top3_pair_hit}/{n}  (chance 3/10)")
    print("\nWhy the colour ranking is so weak — signal vs noise:")
    print(f"  real spread between colours (sd)   {np.mean(spreads):.3f}")
    print(f"  our error on a colour's score      {np.mean(errs):.3f}"
          f"   -> error is {np.mean(errs)/np.mean(spreads):.1f}x the spread")
    print(f"  real spread between pairs (sd)     {np.mean(pspreads):.3f}")
    print(f"  our error on a pair's score        {np.mean(perrs):.3f}"
          f"   -> error is {np.mean(perrs)/np.mean(pspreads):.1f}x the spread")
    return per_set


if __name__ == "__main__":
    run()
