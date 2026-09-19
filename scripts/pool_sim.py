"""The only question that matters at a prerelease: how good is the deck you build?

Grade agreement is a harsh proxy. Mixing up C and C+ counts as a miss but changes
nothing you do. So simulate the actual task: open six packs, build the best 23
spells in two colours using the model's ranking, and compare that deck to the
best deck that pool could have produced.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import predict

ROOT = os.path.join(os.path.dirname(__file__), "..")
WUBRG = "WUBRG"
PAIRS = [a + b for i, a in enumerate(WUBRG) for b in WUBRG[i + 1:]]
# a play booster is roughly 7 commons, 3 uncommons, 1 rare/mythic of playable slots
PACK = {"common": 7, "uncommon": 3, "rare": 1}
PACKS = 6
SPELLS = 23


def build_deck(pool, score):
    """Best two-colour deck by a given scoring function. Returns chosen indices."""
    best, best_val = None, -1e9
    for pair in PAIRS:
        ok = [i for i in pool
              if set(cards[i].get("color_identity") or []) <= set(pair)]
        if len(ok) < SPELLS:
            continue
        ok.sort(key=lambda i: -score[i])
        val = float(np.mean([score[i] for i in ok[:SPELLS]]))
        if val > best_val:
            best, best_val = ok[:SPELLS], val
    return best


def run(setcode, model_z, truth_z, n_pools=400, seed=0):
    rng = np.random.default_rng(seed)
    by_rar = {}
    for i, c in enumerate(cards):
        r = c["rarity"]
        by_rar.setdefault("rare" if r in ("rare", "mythic") else r, []).append(i)
    regrets, opt_vals, got_vals = [], [], []
    for _ in range(n_pools):
        pool = []
        for _p in range(PACKS):
            for rar, k in PACK.items():
                avail = by_rar.get(rar, [])
                if len(avail) >= k:
                    pool += list(rng.choice(avail, k, replace=False))
        pool = list(dict.fromkeys(pool))
        mine = build_deck(pool, model_z)
        best = build_deck(pool, truth_z)
        if not mine or not best:
            continue
        got = float(np.mean([truth_z[i] for i in mine]))
        opt = float(np.mean([truth_z[i] for i in best]))
        got_vals.append(got); opt_vals.append(opt); regrets.append(opt - got)
    return np.array(regrets), np.array(opt_vals), np.array(got_vals)


if __name__ == "__main__":
    import train, dataset, textfeat
    setcode = sys.argv[1] if len(sys.argv) > 1 else "BLB"
    cards = json.load(open(f"{ROOT}/data/cards/{setcode}.json"))
    cards = [c for c in cards if "Basic Land" not in c.get("type_line", "")]

    # honest setup: model that never saw this set
    sets = [s for s in train.TRAIN_SETS
            if os.path.exists(f"{ROOT}/data/labels/{s}.json") and s != setcode]
    X, y, texts, w, rows = dataset.build(sets, verbose=False)
    vec = textfeat.vectorizer()
    Xtr = train.combine(X, texts, vec, fit_vec=True)
    models = train.fit_ensemble(Xtr, y, w, train.feature_names(vec))

    fc = dataset.set_features(setcode)
    import textfeat as tf
    Xh = np.array([[fc[c["name"]].get(k, np.nan) for k in dataset.FEATURE_KEYS] for c in cards])
    Xhf = train.combine(Xh, [tf.normalize(c) for c in cards], vec)
    model_z = predict.rank_to_z(train.predict_ensemble(models, Xhf))

    lab = json.load(open(f"{ROOT}/data/labels/{setcode}.json"))["cards"]
    wr = []
    for c in cards:
        rec = next((lab[v] for v in dataset.name_variants(c) if v in lab), None)
        wr.append(rec["all"]["gih_wr"] if rec and rec["all"]["gih_wr"] else np.nan)
    wr = np.array(wr, dtype=float)
    med = np.nanmedian(wr); wr[~np.isfinite(wr)] = med
    truth_z = predict.rank_to_z(wr)

    rng = np.random.default_rng(0)
    reg, opt, got = run(setcode, model_z, truth_z)
    rnd, _, gotr = run(setcode, rng.normal(0, 1, len(cards)), truth_z, seed=1)
    print(f"{setcode}: {len(reg)} simulated six-pack pools, model never saw this set\n")
    print(f"  best deck in the pool      mean card quality {opt.mean():+.3f} z")
    print(f"  deck the model builds      {got.mean():+.3f} z   (gives up {reg.mean():.3f})")
    print(f"  deck from random picks     {gotr.mean():+.3f} z")
    span = opt.mean() - gotr.mean()
    print(f"\n  model captures {100*(got.mean()-gotr.mean())/span:.0f}% of the gap "
          f"between random and perfect")
    print(f"  pools where the model's deck is within 0.1z of optimal: "
          f"{100*np.mean(reg < 0.1):.0f}%")
