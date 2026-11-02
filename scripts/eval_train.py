"""Train a "who wins from here?" model on simulated positions (sim/SimData.java).

    eval_train.py <positions.tsv.gz> <out_model.json>

The point is to replace XMage's hand-written position score inside its
look-ahead, so the bar is XMage's own score: on held-out GAMES (positions from
one game are near-duplicates, so splitting by row would flatter everything),
does the learned model predict the winner better than that score does?

The model is exported as plain weights -- a standardiser and a small MLP -- so a
few lines of Java (sim/LearnedEval.java) evaluate it identically.
"""
import gzip, io, json, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def read_partial_gz(path):
    """The logger is still appending, so the gzip stream may end mid-member."""
    buf = io.BytesIO()
    with gzip.open(path, "rb") as f:
        try:
            while chunk := f.read(1 << 20):
                buf.write(chunk)
        except (EOFError, OSError):
            pass
    text = buf.getvalue().decode("utf8", "replace")
    text = text[: text.rfind("\n") + 1]
    return pd.read_csv(io.StringIO(text), sep="\t")


def score(name, p, y, turns):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    out = {"model": name, "logloss": round(log_loss(y, p), 4), "auc": round(roc_auc_score(y, p), 4)}
    for lo, hi in ((1, 4), (5, 8), (9, 99)):
        m = (turns >= lo) & (turns <= hi)
        if m.sum() > 100 and len(set(y[m])) == 2:
            out[f"auc_t{lo}-{hi if hi < 99 else '+'}"] = round(roc_auc_score(y[m], p[m]), 4)
    print("  " + "  ".join(f"{k}={v}" for k, v in out.items()), flush=True)
    return out


if __name__ == "__main__":
    src, out_path = sys.argv[1], sys.argv[2]
    df = read_partial_gz(src)
    df = df[df["won"] != 0.5]
    # the last game may be cut mid-write; drop it
    df = df[df["game"] != df["game"].iloc[-1]]
    feats = [c for c in df.columns if c not in ("game", "deckA", "deckB", "seat", "step", "won")]
    games = df["game"].unique()
    rng = np.random.default_rng(0)
    test_games = set(rng.choice(games, size=len(games) // 5, replace=False))
    te = df["game"].isin(test_games).values
    X, y, turns = df[feats].values.astype(float), df["won"].values.astype(int), df["turn"].values
    print(f"{len(games)} games, {len(df)} positions ({te.sum()} held out from {len(test_games)} games), "
          f"{len(feats)} features\n")

    xi = feats.index("xmage_score")
    life = (df["me_life"] - df["opp_life"]).values[:, None].astype(float)
    res = []
    lr1 = LogisticRegression().fit(X[~te][:, [xi]] / 100, y[~te])
    res.append(score("XMage's own score", lr1.predict_proba(X[te][:, [xi]] / 100)[:, 1], y[te], turns[te]))
    lr2 = LogisticRegression().fit(life[~te], y[~te])
    res.append(score("life difference only", lr2.predict_proba(life[te])[:, 1], y[te], turns[te]))

    sc = StandardScaler().fit(X[~te])
    Xs_tr, Xs_te = sc.transform(X[~te]), sc.transform(X[te])
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xs_tr, y[~te])
    res.append(score("logistic, all features", lr.predict_proba(Xs_te)[:, 1], y[te], turns[te]))
    mlp = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=200, early_stopping=True,
                        random_state=0).fit(Xs_tr, y[~te])
    res.append(score("MLP 32, all features", mlp.predict_proba(Xs_te)[:, 1], y[te], turns[te]))

    best = min(res[2:], key=lambda r: r["logloss"])
    if best["model"].startswith("MLP"):
        layers = [{"W": w.T.tolist(), "b": b.tolist()} for w, b in zip(mlp.coefs_, mlp.intercepts_)]
        act = "relu"
    else:
        layers = [{"W": lr.coef_.tolist(), "b": lr.intercept_.tolist()}]
        act = "none"
    json.dump({"features": feats, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
               "layers": layers, "hidden_activation": act, "trained_on_games": int(len(games)),
               "results": res}, open(out_path, "w"))
    print(f"\nexported {best['model']} -> {out_path}")
