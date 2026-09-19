"""Trace one card end to end, so the number is auditable rather than trusted.

Prints every step between a Scryfall card and a letter grade.
"""
import json, os, sys, pickle
import numpy as np
import scipy.sparse as sp
from scipy.stats import norm
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
import dataset, features, textfeat, guards, predict, train as trainmod

ROOT = os.path.join(os.path.dirname(__file__), "..")


def explain(name, setcode="FRA"):
    cards = [c for c in json.load(open(f"{ROOT}/data/cards/{setcode}.json"))
             if "Basic Land" not in c.get("type_line", "")]
    card = next((c for c in cards if c["name"] == name
                 or c["name"].split(" // ")[0] == name
                 or c["name"].split(",")[0] == name), None)
    if card is None:
        raise SystemExit(f"no card named {name!r} in {setcode}")

    model = lgb.Booster(model_file=f"{ROOT}/data/model.txt")
    vec = pickle.load(open(f"{ROOT}/data/vectorizer.pkl", "rb"))
    ridge = pickle.load(open(f"{ROOT}/data/ridge.pkl", "rb"))

    print(f"CARD   {card['name']}  {card.get('mana_cost','')}  [{card['rarity']}]")
    print(f"       {card.get('type_line','')}")

    # 1 -- numeric features
    f = features.extract(card)
    print(f"\n1. FEATURES ({len(f)} numeric, set-agnostic)")
    live = {k: v for k, v in f.items() if v not in (0.0, None) and not (isinstance(v, float) and np.isnan(v))}
    for k in sorted(live)[:14]:
        print(f"     {k:22s} {live[k]}")
    print(f"     ... {len(live)-14} more non-zero" if len(live) > 14 else "")

    # 2 -- text
    norm_txt = textfeat.normalize(card)
    print(f"\n2. ORACLE TEXT normalised (reminder text stripped, name -> ~, numbers -> numval)")
    print(f"     \"{norm_txt[:150]}\"")
    T = vec.transform([norm_txt])
    nz = T.nonzero()[1]
    vocab = vec.get_feature_names_out()
    top = sorted(((T[0, j], vocab[j]) for j in nz), reverse=True)[:8]
    print(f"     -> {len(nz)} of {len(vocab)} vocabulary terms fire; strongest:")
    print("        " + ", ".join(f"{t}({w:.2f})" for w, t in top))

    # 3 -- model
    X = np.array([[f.get(k, np.nan) for k in dataset.FEATURE_KEYS]])
    Xf = trainmod.combine(X, [norm_txt], vec)
    nv = trainmod.novelty([norm_txt], vec)[0]
    print(f"\n3. MODEL  ensemble over {Xf.shape[1]} inputs")
    print(f"     GBM (weight {trainmod.ENSEMBLE_GBM_WEIGHT}) + ridge, rank-averaged")
    print(f"     wording novelty: oov {nv[0]:.2f}, mean term rarity {nv[1]:.2f}")

    # 4 -- rank within set
    allX = np.array([[features.extract(c).get(k, np.nan) for k in dataset.FEATURE_KEYS]
                     for c in cards])
    alltxt = [textfeat.normalize(c) for c in cards]
    allF = trainmod.combine(allX, alltxt, vec)
    allraw = trainmod.predict_ensemble((model, ridge), allF)
    raw = float(allraw[[c["name"] for c in cards].index(card["name"])])
    rank = int((allraw < raw).sum())
    print(f"     -> rank-averaged score {raw:.4f}")
    z = predict.rank_to_z(allraw)[[c["name"] for c in cards].index(card["name"])]
    print(f"\n4. CURVE  rank {rank+1} of {len(cards)} by raw score")
    print(f"     rank -> quantile {(rank+0.5)/len(cards):.4f} -> z {z:+.4f}")
    print(f"     (only the ORDER of raw scores matters; magnitudes are discarded here)")

    # 5 -- grade
    curve = predict.empirical_grade_curve()
    letters = predict.assign_grades(allraw, curve)
    grade = letters[[c["name"] for c in cards].index(card["name"])]
    cum, cut = 0.0, None
    for g, frac in curve:
        cum += frac
        if g == grade:
            cut = cum
            break
    print(f"\n5. GRADE  rank {rank+1}/{len(cards)} mapped onto the grade distribution that")
    print(f"     released sets actually have (not a forced normal -- real win rates are")
    print(f"     fat-tailed, so a normal under-produces A+):")
    print("        " + "  ".join(f"{g}:{f*100:.1f}%" for g, f in curve[:7]))
    print("        " + "  ".join(f"{g}:{f*100:.1f}%" for g, f in curve[7:]))
    print(f"     -> top {cut*100:.1f}% of the set gets {grade} or better  ->  {grade}")

    # 6 -- guards (advisory only)
    calib = json.load(open(f"{ROOT}/data/calibration.json")) if os.path.exists(
        f"{ROOT}/data/calibration.json") else None
    if calib:
        for i, b in enumerate(calib["bins"]):
            lo = -np.inf if b["lo"] is None else b["lo"]
            hi = np.inf if b["hi"] is None else b["hi"]
            if lo <= nv[1] < hi:
                print(f"\n5b. TRUST  wording familiarity band {i+1} of {len(calib['bins'])}")
                print(f"     cards in this band missed their real grade by "
                      f"{b['mean_grade_error']:.2f} steps on average "
                      f"({b['within_1']:.0%} within one)")
                break
    az, adj, flags, conf = guards.apply(card, setcode, z)
    print(f"\n6. GUARDS  adjustments {'ON' if guards.APPLY_ADJUSTMENTS else 'OFF (advisory only)'}"
          f"  ->  grade stays {grade}")
    print(f"     confidence: {conf}")
    for a in adj:
        print(f"     note: {a['reason']}")
    for fl in flags:
        print(f"     flag: {fl['desc']} — {fl['n_precedent']} precedents"
              + (f", mean {fl['precedent_mean_z']:+.2f}z" if fl['precedent_mean_z'] is not None else ""))
    print(f"\nFINAL  {card['name']}: {grade}")


if __name__ == "__main__":
    explain(sys.argv[1] if len(sys.argv) > 1 else "Last Gasp",
            sys.argv[2] if len(sys.argv) > 2 else "FRA")
