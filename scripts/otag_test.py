"""Do Scryfall's curated oracle tags add anything our own features do not?

Every feature in features.py is something we derived from the card ourselves.
These are not: a human-curated functional taxonomy that catches what a regex
misses. New information rather than a cleverer reading of what we already had,
which is the only kind of change that has ever worked here.
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr, ttest_rel, wilcoxon
sys.path.insert(0, os.path.dirname(__file__))
import dataset, textfeat, train, fetch_otags

ROOT = os.path.join(os.path.dirname(__file__), "..")
tags = json.load(open(f"{ROOT}/data/otags.json"))
TAGS = fetch_otags.TAGS
OTNAMES = [f"ot_{t.replace('-', '_')}" for t in TAGS] + ["ot_count"]

sets = [s for s in train.TRAIN_SETS if os.path.exists(f"{ROOT}/data/labels/{s}.json")]
X, y, texts, w, rows = dataset.build(sets, verbose=False)
arr = np.array([r["set"] for r in rows])

def tagvec(name):
    t = tags.get(name) or tags.get(name.split(" // ")[0]) or []
    return [1.0 if g in t else 0.0 for g in TAGS] + [float(len(t))]
OT = np.array([tagvec(r["name"]) for r in rows])
print(f"{len(rows)} cards, {(OT[:, :-1].sum(1) > 0).mean():.0%} carry at least one tag\n",
      flush=True)
ei = dataset.FEATURE_KEYS.index("expert_rating")

def namer(extra):
    def f(vec):
        return [n.replace(" ", "_") for n in
                dataset.FEATURE_KEYS + extra + ["oov_rate", "mean_idf"]
                + [f"tf_{g}" for g in vec.get_feature_names_out()]]
    return f

def sweep(Xa, names):
    out = {}
    for s in sets:
        te = arr == s
        if te.sum() < 30:
            continue
        tr = ~te
        vec = textfeat.vectorizer()
        Xtr = train.combine(Xa[tr], [t for t, m in zip(texts, tr) if m], vec, fit_vec=True)
        Xte = train.combine(Xa[te], [t for t, m in zip(texts, te) if m], vec)
        out[s] = float(spearmanr(train.predict_ensemble(
            train.fit_ensemble(Xtr, y[tr], w[tr], names(vec)), Xte), y[te]).statistic)
    return out

res = {}
for mode in ("blind", "full"):
    Xb = X.copy()
    if mode == "blind":
        Xb[:, ei] = np.nan
    base = sweep(Xb, namer([]))
    with_ot = sweep(np.hstack([Xb, OT]), namer(OTNAMES))
    ks = sorted(set(base) & set(with_ot))
    a = np.array([base[k] for k in ks]); b = np.array([with_ot[k] for k in ks])
    d = b - a
    print(f"[{mode}] baseline {a.mean():.4f}   with oracle tags {b.mean():.4f}   ({d.mean():+.4f})")
    print(f"        improved {int((d>0).sum())}/{len(d)}  t={ttest_rel(b,a).pvalue:.4f}  "
          f"w={wilcoxon(b,a).pvalue:.4f}", flush=True)
    res[mode] = {"base": base, "otags": with_ot, "delta": float(d.mean()),
                 "p": float(ttest_rel(b, a).pvalue)}
json.dump(res, open(f"{ROOT}/data/otag_test.json", "w"), indent=1)
print("\nDONE")
