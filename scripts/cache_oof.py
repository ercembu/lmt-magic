"""Cache out-of-fold predictions so diagnostics don't refit 21 models each time."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import dataset, textfeat, train

ROOT = os.path.join(os.path.dirname(__file__), "..")

if __name__ == "__main__":
    sets = [s for s in train.TRAIN_SETS
            if os.path.exists(f"{ROOT}/data/labels/{s}.json")]
    X, y, texts, w, rows = dataset.build(sets, verbose=False)
    arr = np.array([r["set"] for r in rows])
    pred = np.zeros(len(rows)); nov = np.zeros(len(rows))
    for s in sets:
        te = arr == s
        if te.sum() < 30:
            continue
        tr = ~te
        vec = textfeat.vectorizer()
        Xtr = train.combine(X[tr], [t for t, m in zip(texts, tr) if m], vec, fit_vec=True)
        tt = [t for t, m in zip(texts, te) if m]
        Xte = train.combine(X[te], tt, vec)
        pred[te] = train.predict_ensemble(
            train.fit_ensemble(Xtr, y[tr], w[tr], train.feature_names(vec)), Xte)
        nov[te] = train.novelty(tt, vec)[:, 1]
        print(f"  {s} done", flush=True)
    out = [{"set": r["set"], "name": r["name"], "y": float(r["y"]),
            "gih_wr": r["gih_wr"], "gih_n": r["gih_n"], "pred": float(p),
            "novelty": float(nv),
            "colors": r["card"].get("colors") or [],
            "color_identity": r["card"].get("color_identity") or [],
            "rarity": r["card"]["rarity"], "cmc": r["card"].get("cmc", 0),
            "type_line": r["card"].get("type_line", "")}
           for r, p, nv in zip(rows, pred, nov)]
    json.dump(out, open(f"{ROOT}/data/oof.json", "w"))
    print(f"cached {len(out)} out-of-fold predictions")
