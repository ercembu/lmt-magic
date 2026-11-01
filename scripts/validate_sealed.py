"""Does a model trained on Sealed data predict Sealed better than the draft one?

The whole project is trained on Premier Draft win rates because that is where
the data is. But the use case is a prerelease, which is Sealed: bigger decks,
worse mana, slower games, and removal that cannot be sided out. If the two
formats rank cards differently, we have been answering the wrong question.
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr, wilcoxon
sys.path.insert(0, "scripts")
import dataset, textfeat, train

SEALED = [s for s in train.TRAIN_SETS
          if os.path.exists(f"data/labels/{s}.Sealed.json")
          and os.path.exists(f"data/labels/{s}.json")]
print(f"{len(SEALED)} sets with both draft and sealed labels: {' '.join(SEALED)}\n")

# how differently do the two formats even rank cards?
same = []
for s in SEALED:
    d = {r["name"]: r for r in dataset.load_set(s, fmt="PremierDraft")}
    e = {r["name"]: r for r in dataset.load_set(s, fmt="Sealed")}
    n = [k for k in d if k in e and d[k]["gih_wr"] and e[k]["gih_wr"]
         and d[k]["gih_n"] >= 400 and e[k]["gih_n"] >= 400]
    if len(n) >= 40:
        same.append(spearmanr([d[k]["gih_wr"] for k in n],
                              [e[k]["gih_wr"] for k in n]).statistic)
print(f"draft vs sealed card rankings agree at {np.mean(same):.3f} "
      f"(range {min(same):.2f}-{max(same):.2f}) -- the formats are close but not identical\n")

Xd, yd, td, wd, rd = dataset.build(SEALED, verbose=False, fmt="PremierDraft")
Xs, ys, ts, ws, rs = dataset.build(SEALED, verbose=False, fmt="Sealed")
ad = np.array([r["set"] for r in rd]); asx = np.array([r["set"] for r in rs])

res = {}
for s in SEALED:
    te = asx == s
    if te.sum() < 30:
        continue
    row = {}
    for label, (X, y, t, w, arr) in (("draft", (Xd, yd, td, wd, ad)),
                                     ("sealed", (Xs, ys, ts, ws, asx))):
        tr = arr != s
        vec = textfeat.vectorizer()
        Xtr = train.combine(X[tr], [x for x, m in zip(t, tr) if m], vec, fit_vec=True)
        Xte = train.combine(Xs[te], [x for x, m in zip(ts, te) if m], vec)
        p = train.predict_ensemble(
            train.fit_ensemble(Xtr, y[tr], w[tr], train.feature_names(vec)), Xte)
        row[label] = float(spearmanr(p, ys[te]).statistic)
    res[s] = row
    print(f"  {s}  trained on draft {row['draft']:.4f}   on sealed {row['sealed']:.4f}   "
          f"{row['sealed']-row['draft']:+.4f}", flush=True)

a = np.array([r["draft"] for r in res.values()])
b = np.array([r["sealed"] for r in res.values()])
print(f"\nscored on SEALED, trained on draft labels  {a.mean():.4f}")
print(f"scored on SEALED, trained on sealed labels {b.mean():.4f}   ({b.mean()-a.mean():+.4f})")
print(f"sets improved {int((b>a).sum())}/{len(b)}   wilcoxon p={wilcoxon(a,b).pvalue:.4f}")
