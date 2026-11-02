"""Turn simulated games into a value per card, and check it against reality.

Each game is a pair of 40-card decks and a winner. Model:

    P(A beats B) = logistic( sum of A's card values - sum of B's )

fitted as a ridge-penalised logistic regression on the card-count difference,
so a card seen in few decks stays near zero rather than taking a wild value.
That is the same idea as 17lands' "game in deck" win rate, but it credits a
card for its own contribution instead of for the company it keeps.

Scored against FRA's real 17lands win rates (Sealed first -- the format a
prerelease is -- then Premier Draft), and against the blind model, because the
only question that matters is whether this is information the model lacks.
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_games(results, manifest):
    decks = {m["deck"]: m["cards"] for m in json.load(open(manifest))}
    games = []
    for line in open(results):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3 or p[2] not in ("A", "B"):
            continue
        a, b = (os.path.basename(x)[:-4] for x in p[:2])
        games.append((decks[a], decks[b], p[2] == "A"))
    return games


def fit(games, C=0.5):
    names = sorted({c for a, b, _ in games for c in a + b})
    ix = {n: i for i, n in enumerate(names)}
    X = np.zeros((2 * len(games), len(names)))
    y = np.zeros(2 * len(games))
    for g, (a, b, a_won) in enumerate(games):
        for c in a: X[2 * g, ix[c]] += 1
        for c in b: X[2 * g, ix[c]] -= 1
        X[2 * g + 1] = -X[2 * g]           # both orientations: no seat bias to learn
        y[2 * g], y[2 * g + 1] = a_won, not a_won
    m = LogisticRegression(C=C, fit_intercept=False, max_iter=2000).fit(X, y)
    seen = (X[::2] != 0).sum(0)
    return {n: (float(m.coef_[0, i]), int(seen[i])) for n, i in ix.items()}


def truth(setcode, fmt):
    suffix = "" if fmt == "PremierDraft" else f".{fmt}"
    lab = json.load(open(f"{ROOT}/data/labels/{setcode}{suffix}.json"))["cards"]
    return {n: r["all"]["gih_wr"] for n, r in lab.items()
            if r["all"]["gih_wr"] and r["all"]["gih_n"] >= 400}


if __name__ == "__main__":
    results, manifest = sys.argv[1], sys.argv[2]
    setcode = sys.argv[3] if len(sys.argv) > 3 else "FRA"
    min_seen = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    games = load_games(results, manifest)
    print(f"{len(games)} decided games")
    vals = fit(games)
    json.dump(vals, open(f"{os.path.dirname(results)}/card_values.json", "w"), indent=1)
    for fmt in ("Sealed", "PremierDraft"):
        t = truth(setcode, fmt)
        ks = [k for k, (v, n) in vals.items() if k in t and n >= min_seen]
        rho = spearmanr([vals[k][0] for k in ks], [t[k] for k in ks]).statistic
        print(f"  simulated value vs real {fmt:12s} GIH WR: spearman {rho:+.3f}  on {len(ks)} cards")
    # does it carry information the blind model lacks?
    pre = {c["name"]: c for c in json.load(open(f"{ROOT}/data/grades_{setcode}.prerelease.json"))}
    t = truth(setcode, "Sealed")
    ks = [k for k, (v, n) in vals.items() if k in t and k in pre and n >= min_seen]
    s = np.array([vals[k][0] for k in ks]); m = np.array([pre[k]["raw_score"] for k in ks])
    r = np.array([t[k] for k in ks])
    from scipy.stats import rankdata
    zs, zm, zr = (rankdata(v) / len(v) for v in (s, m, r))
    print(f"  shipped model alone {spearmanr(zm, zr).statistic:+.3f}   sim alone "
          f"{spearmanr(zs, zr).statistic:+.3f}   sim vs model {spearmanr(zs, zm).statistic:+.3f}")
    for wgt in (0.1, 0.2, 0.3):
        print(f"  model + {wgt:.0%} sim: {spearmanr((1 - wgt) * zm + wgt * zs, zr).statistic:+.3f}")
