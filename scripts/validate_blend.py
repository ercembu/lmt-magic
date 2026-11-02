"""Average the two formats' win rates into one target.

Training on sealed instead of draft failed (-0.007): sealed dumps are a fifth
the size, so the format-specific target costs five times the label noise.
Averaging is a different question. The two formats rank cards at 0.841, so they
are largely two noisy measurements of one quantity -- and averaging two noisy
measurements of the same thing is the oldest variance reduction there is.

This is "more information", not cleverer processing of the same information,
which is the only kind of change that has ever worked in this project.
"""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr, ttest_rel, wilcoxon
sys.path.insert(0, "scripts")
import dataset, textfeat, train

sets = [s for s in train.TRAIN_SETS if os.path.exists(f"data/labels/{s}.json")]
X, y, texts, w, rows = dataset.build(sets, verbose=False)
arr = np.array([r["set"] for r in rows])

# sealed z for the same cards, where the set has a sealed dump
sealed = {}
for s in sets:
    if not os.path.exists(f"data/labels/{s}.Sealed.json"):
        continue
    rs = [r for r in dataset.load_set(s, fmt="Sealed")
          if r["gih_wr"] and r["gih_n"] >= 400]
    if len(rs) < 40:
        continue
    v = np.array([r["gih_wr"] for r in rs])
    mu, sd = v.mean(), v.std()
    for r in rs:
        sealed[(s, r["name"])] = (r["gih_wr"] - mu) / sd
print(f"{len(sets)} sets, sealed z for {len(sealed)} card-set pairs "
      f"({len({k[0] for k in sealed})} sets)\n")

def blended(lam):
    yy = y.copy()
    hit = 0
    for i, r in enumerate(rows):
        sz = sealed.get((r["set"], r["name"]))
        if sz is not None:
            yy[i] = (1 - lam) * y[i] + lam * sz
            hit += 1
    return yy, hit

def sweep(yy):
    out = {}
    for s in sets:
        te = arr == s
        if te.sum() < 30:
            continue
        tr = ~te
        vec = textfeat.vectorizer()
        Xtr = train.combine(X[tr], [t for t, m in zip(texts, tr) if m], vec, fit_vec=True)
        Xte = train.combine(X[te], [t for t, m in zip(texts, te) if m], vec)
        p = train.predict_ensemble(
            train.fit_ensemble(Xtr, yy[tr], w[tr], train.feature_names(vec)), Xte)
        # always scored against the draft target, so the comparison is fair
        out[s] = float(spearmanr(p, y[te]).statistic)
    return out

base = sweep(y)
print(f"  {'target':30s} {'mean':>6s} {'delta':>8s}  won   t-test")
print(f"  {'draft only (today)':30s} {np.mean(list(base.values())):.4f}")
res = {"base": base}
for lam in (0.25, 0.4, 0.5):
    yy, hit = blended(lam)
    r = sweep(yy)
    ks = sorted(set(base) & set(r))
    a = np.array([base[k] for k in ks]); b = np.array([r[k] for k in ks])
    d = b - a
    print(f"  {f'{1-lam:.2f} draft / {lam:.2f} sealed':30s} {b.mean():.4f} {d.mean():+8.4f}  "
          f"{int((d>0).sum())}/{len(d)}  p={ttest_rel(b,a).pvalue:.4f}"
          + ("   KEEP" if d.mean() > 0 and ttest_rel(b, a).pvalue < 0.05 else ""), flush=True)
    res[str(lam)] = r
json.dump(res, open(f"{os.environ['SC']}/blend.json", "w"), indent=1)
