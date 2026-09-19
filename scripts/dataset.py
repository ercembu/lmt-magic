"""Join Scryfall card data to 17lands performance labels, per set."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import features, textfeat, context

ROOT = os.path.join(os.path.dirname(__file__), "..")
MIN_GAMES = 400          # below this the win rate is mostly noise

_sample_path = f"{ROOT}/data/cards/FRA.json"
if not os.path.exists(_sample_path):
    raise SystemExit("No card data yet — run scripts/fetch_scryfall.py first.")
_sample = json.load(open(_sample_path))[0]
BASE_KEYS = sorted(features.extract(_sample))
# Set-context and per-colour-context features were built and tested (context.py).
# Neither survived: set context gave +0.002 spearman (p=0.36), and per-colour
# context was actively harmful (-0.011 card-level, colour-depth 0.13 -> 0.06,
# colour error clustering 1.3x -> 1.59x). The second result is structural rather
# than bad luck: a feature that is constant within a colour can only ever add
# colour-correlated error, which is the exact thing it was meant to remove.
# context.py is kept so the experiment is reproducible; it is not used.
# Published pre-release expert ratings (expert_grades.py). The only signal here
# that is not derived from card text, and the largest single gain measured:
# +0.076 spearman. NaN when no review has been published for a set, which is the
# normal state for the first days of preview season -- the model then falls back
# to text and stats alone.
FEATURE_KEYS = BASE_KEYS + ["expert_rating"]


def set_features(setcode):
    """Per-card feature dicts for a whole set.

    Includes the published expert rating where a set review exists. Reviews come
    out during preview season, so this is available for an unreleased set in a
    way win rates never are -- but only once someone has published one.
    """
    cards = json.load(open(f"{ROOT}/data/cards/{setcode}.json"))
    base = [features.extract(c) for c in cards]
    ep = f"{ROOT}/data/expert/{setcode}.json"
    expert = json.load(open(ep)) if os.path.exists(ep) else {}
    out = {}
    for c, f in zip(cards, base):
        f["expert_rating"] = float(expert.get(c["name"], np.nan))
        out[c["name"]] = f
    return out


def name_variants(card):
    """17lands names cards slightly differently than Scryfall for DFCs/split."""
    n = card["name"]
    out = {n, n.split(" // ")[0]}
    if "card_faces" in card:
        out |= {f["name"] for f in card["card_faces"]}
        out.add(" // ".join(f["name"] for f in card["card_faces"]))
    return out


def load_set(setcode, group="all", fmt="PremierDraft"):
    cards = json.load(open(f"{ROOT}/data/cards/{setcode}.json"))
    suffix = "" if fmt == "PremierDraft" else f".{fmt}"
    lp = f"{ROOT}/data/labels/{setcode}{suffix}.json"
    labels = json.load(open(lp))["cards"] if os.path.exists(lp) else {}
    rows = []
    for c in cards:
        rec = next((labels[v] for v in name_variants(c) if v in labels), None)
        y, n = (None, 0)
        if rec:
            y, n = rec[group]["gih_wr"], rec[group]["gih_n"]
        rows.append({"set": setcode, "name": c["name"], "card": c,
                     "gih_wr": y, "gih_n": n})
    return rows


def build(setcodes, group="all", min_games=MIN_GAMES, verbose=True, fmt="PremierDraft"):
    """X (dense features), y (within-set z-scored GIH WR), texts, weights, rows."""
    allrows = []
    for s in setcodes:
        rows = [r for r in load_set(s, group, fmt)
                if r["gih_wr"] is not None and r["gih_n"] >= min_games]
        if not rows:
            continue
        wr = np.array([r["gih_wr"] for r in rows])
        mu, sd = wr.mean(), wr.std()
        for r, w in zip(rows, wr):
            r["y"], r["set_mu"], r["set_sd"] = (w - mu) / sd, mu, sd
        allrows += rows
        if verbose:
            print(f"  {s}: {len(rows):3d} cards  mean GIH {mu:.4f}  sd {sd:.4f}")
    feat_cache = {s: set_features(s) for s in {r["set"] for r in allrows}}
    X = np.array([[feat_cache[r["set"]][r["name"]].get(k, np.nan) for k in FEATURE_KEYS]
                  for r in allrows])
    y = np.array([r["y"] for r in allrows])
    texts = [textfeat.normalize(r["card"]) for r in allrows]
    # More games behind a win rate means a more trustworthy label.
    w = np.sqrt(np.array([r["gih_n"] for r in allrows], dtype=float))
    w = w / w.mean()
    return X, y, texts, w, allrows
