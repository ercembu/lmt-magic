"""A tag graph: which kinds of card get better when the set is full of which others.

Every feature in the model describes a card on its own. Limited is not played
that way -- a sacrifice outlet is a different card in a set full of death
triggers. This learns that from the 32 released sets as a graph: nodes are what
a card does (Scryfall's oracle tags, our effect/keyword/trigger/type flags, and
a handful of generic payoff patterns); an edge A -> B is how much a card that
does A gains from the *supply* of B among the cards it can be played beside
(same colour, or colourless), rarity-weighted because commons are what you
actually open.

The per-colour context features failed (dataset.py) because they were constant
across a colour, so their error hit every card of that colour at once. A card's
synergy score is its own nodes times its colours' supply, so it varies card by
card and does not share that failure mode -- whether it predicts anything is the
question this answers.

    synergy.py screen   ridge with/without the edges, leave-one-set-out
    synergy.py graph    fit on every set and print the strongest edges
"""
import json, os, re, sys
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr, ttest_rel
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(__file__))
import dataset, features, train

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Generic payoff patterns. The flags in features.py mostly say what a card *does*;
# synergy needs what it *wants*.
PAYOFFS = {
    "wants_deaths":    r"whenever (another |a )?(nontoken )?creature( you control)? dies",
    "wants_artifacts": r"for each artifact|artifacts? you control|whenever an(other)? artifact",
    "wants_spells":    r"whenever you cast an? (instant|sorcery|noncreature)",
    "wants_graveyard": r"cards? in your graveyard|from your graveyard",
    "wants_etb":       r"whenever another (nontoken )?creature( you control)? enters",
    "wants_counters":  r"with a \+1/\+1 counter|for each \+1/\+1 counter|counters? on (it|them)",
    "wants_wide":      r"for each (other )?creature you control|creatures you control get",
    "wants_lifegain":  r"whenever you gain life",
    "wants_enchant":   r"enchantment spell|for each (aura|enchantment)|enchanted creature",
    "wants_legends":   r"legendary (creature|spell|permanent)s?",
    "wants_tokens":    r"whenever .{0,20}token|tokens? you control",
}
FLAG_PREFIXES = ("e_", "kw_", "trig_", "type_")
SKIP = {"e_removal_quality", "type_land"}
RARITY_W = {"common": 1.0, "uncommon": 0.3, "rare": 0.07, "mythic": 0.035}


def node_names():
    flags = [k for k in dataset.BASE_KEYS if k.startswith(FLAG_PREFIXES) and k not in SKIP]
    return flags + [f"ot_{t.replace('-', '_')}" for t in dataset.OTAG_VOCAB] + list(PAYOFFS)


NODES = node_names()


def nodes(card):
    f = features.extract(card)
    f.update(dataset.otag_features(card["name"]))
    t = features.full_text(card).lower() if hasattr(features, "full_text") else (
        (card.get("oracle_text") or "") + " ".join(x.get("oracle_text", "") or ""
                                                    for x in card.get("card_faces", []))).lower()
    for k, p in PAYOFFS.items():
        f[k] = 1.0 if re.search(p, t) else 0.0
    return np.array([1.0 if (f.get(k) or 0) > 0 else 0.0 for k in NODES])


_set_cache = {}


def set_nodes(setcode):
    """Node matrix, colour identity and rarity weight for everything openable in a set."""
    if setcode not in _set_cache:
        cards = [c for c in json.load(open(f"{ROOT}/data/cards/{setcode}.json"))
                 if "Basic Land" not in c.get("type_line", "")]
        N = np.array([nodes(c) for c in cards])
        ci = [set(c.get("color_identity") or []) for c in cards]
        rw = np.array([RARITY_W.get(c["rarity"], 0.1) for c in cards])
        _set_cache[setcode] = ({c["name"]: i for i, c in enumerate(cards)}, N, ci, rw)
    return _set_cache[setcode]


def supply(setcode, name):
    """Rarity-weighted share of each node among the cards this one can be played beside."""
    idx, N, ci, rw = set_nodes(setcode)
    i = idx[name]
    mine = ci[i]
    ok = np.array([j != i and (not mine or not c or bool(mine & c)) for j, c in enumerate(ci)])
    w = rw * ok
    return (w @ N) / w.sum()


def matrices(rows):
    """Card nodes A, partner supply S, and the edge interactions A x S (sparse)."""
    A = np.array([set_nodes(r["set"])[1][set_nodes(r["set"])[0][r["name"]]] for r in rows])
    S = np.array([supply(r["set"], r["name"]) for r in rows])
    S = (S - S.mean(0)) / (S.std(0) + 1e-9)
    k = len(NODES)
    ri, ci, v = [], [], []
    for r_, (a, s) in enumerate(zip(A, S)):
        for ai in np.flatnonzero(a):
            ri += [r_] * k; ci += list(range(ai * k, ai * k + k)); v += list(s)
    E = sp.csr_matrix((v, (ri, ci)), shape=(len(rows), k * k))
    return A, S, E


def base_block(X):
    keep = [dataset.FEATURE_KEYS.index(k) for k in
            ("rarity_ord", "cmc", "power", "toughness", "pt_per_mana", "n_colors")]
    B = np.nan_to_num(X[:, keep])
    return (B - B.mean(0)) / (B.std(0) + 1e-9)


def load():
    sets = [s for s in train.TRAIN_SETS if os.path.exists(f"{ROOT}/data/labels/{s}.json")]
    X, y, texts, w, rows = dataset.build(sets, verbose=False)
    keep = np.array([r["name"] in set_nodes(r["set"])[0] for r in rows])
    rows = [r for r, k in zip(rows, keep) if k]
    return sets, X[keep], y[keep], w[keep], rows


def screen():
    sets, X, y, w, rows = load()
    arr = np.array([r["set"] for r in rows])
    A, S, E = matrices(rows)
    B = base_block(X)
    designs = {
        "card alone (nodes)":          sp.hstack([sp.csr_matrix(B), sp.csr_matrix(A)]).tocsr(),
        "+ supply (set context)":      sp.hstack([sp.csr_matrix(B), sp.csr_matrix(A), sp.csr_matrix(S)]).tocsr(),
        "+ supply + edges (graph)":    sp.hstack([sp.csr_matrix(B), sp.csr_matrix(A), sp.csr_matrix(S), E]).tocsr(),
    }
    res = {}
    for name, D in designs.items():
        for alpha in (10.0, 100.0, 1000.0):
            out = {}
            for s in sets:
                te = arr == s
                if te.sum() < 30:
                    continue
                m = Ridge(alpha=alpha).fit(D[~te], y[~te], sample_weight=w[~te])
                out[s] = spearmanr(m.predict(D[te]), y[te]).statistic
            res[(name, alpha)] = out
            print(f"  {name:28s} alpha {alpha:6.0f}  {np.mean(list(out.values())):.4f}"
                  f"   FRA {out['FRA']:.4f}", flush=True)
    return res


# Unshrunk, the ~10k edge weights memorise each set and the screen collapses
# (0.49 -> 0.23-0.34). Penalising the edges ~1000x harder than the card's own
# nodes -- done by scaling their columns down -- is what makes them transfer.
EDGE_SCALE = 0.03
ALPHA = 10.0


def fit_edges(D_base, E, y, w):
    D = sp.hstack([D_base, E * EDGE_SCALE]).tocsr()
    c = Ridge(alpha=ALPHA).fit(D, y, sample_weight=w).coef_
    k = len(NODES)
    return c[-k * k:].reshape(k, k) * EDGE_SCALE


def graph(top=25, out=f"{ROOT}/data/synergy_graph.json"):
    """Fit on all sets, then refit leaving each set out: an edge is only shown if
    its sign holds in at least 30 of the 32 leave-one-out graphs."""
    sets, X, y, w, rows = load()
    arr = np.array([r["set"] for r in rows])
    A, S, E = matrices(rows)
    Db = sp.csr_matrix(np.hstack([base_block(X), A]))
    W = fit_edges(Db, E, y, w)
    agree = np.zeros_like(W)
    for s in sets:
        m = arr != s
        agree += np.sign(fit_edges(Db[m], E[m], y[m], w[m])) == np.sign(W)
    k, n_a = len(NODES), A.sum(0)
    stable = (agree >= len(sets) - 2) & (n_a[:, None] >= 60)
    order = [f for f in np.argsort(W, axis=None) if stable.flat[f]]
    def show(idxs, label):
        print(f"\n{label}")
        for f in idxs:
            a, b = divmod(int(f), k)
            print(f"  {NODES[a]:24s} <- supply of {NODES[b]:24s} {W[a, b]:+.4f}"
                  f"   (n={int(n_a[a])}, sign held {int(agree[a, b])}/{len(sets)})")
    show(order[::-1][:top], "cards that GAIN from a set full of ...")
    show(order[:top], "cards that LOSE from a set full of ...")
    edges = [{"card": NODES[a], "partner": NODES[b], "w": float(W[a, b]),
              "held": int(agree[a, b]), "n": int(n_a[a])}
             for a, b in zip(*np.nonzero(stable))]
    json.dump({"nodes": NODES, "n_sets": len(sets), "edges": edges}, open(out, "w"), indent=1)
    print(f"\n{stable.sum()} stable edges of {k * k} -> {out}")
    return W


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "screen"
    screen() if cmd == "screen" else graph()
