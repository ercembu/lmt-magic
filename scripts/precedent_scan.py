"""Measure, rather than assume, how much precedent a set's mechanics have.

For each mechanic of the target set, find cards in the 21 training sets whose
text matches the same pattern, and report how those cards actually performed.
This is what decides whether a grade deserves a confidence warning: a mechanic
with 300 precedents in training is one the model understands; a mechanic with
two is one it is guessing about.
"""
import json, os, sys, re
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import dataset, guards, train

ROOT = os.path.join(os.path.dirname(__file__), "..")


def scan(setcode="FRA"):
    sets = [s for s in train.TRAIN_SETS if os.path.exists(f"{ROOT}/data/labels/{s}.json")]
    pool = []
    for s in sets:
        for r in dataset.load_set(s):
            if r["gih_wr"] is not None and r["gih_n"] >= dataset.MIN_GAMES:
                pool.append(r)
    # per-set z so performance is comparable across sets
    for s in sets:
        rs = [r for r in pool if r["set"] == s]
        if not rs:
            continue
        wr = np.array([r["gih_wr"] for r in rs])
        mu, sd = wr.mean(), wr.std()
        for r in rs:
            r["z"] = (r["gih_wr"] - mu) / sd

    out = {}
    for name, m in guards.SET_MECHANICS.get(setcode, {}).items():
        # match older cards on the equivalent EFFECT, not this set's token names
        pat = re.compile(m.get("analogue") or m["pattern"])
        hits = [r for r in pool if pat.search(guards.card_text(r["card"]))]
        zs = np.array([r["z"] for r in hits]) if hits else np.array([])
        by_set = {}
        for r in hits:
            by_set[r["set"]] = by_set.get(r["set"], 0) + 1
        out[name] = {
            "desc": m["desc"],
            "n_precedent": len(hits),
            "sets": dict(sorted(by_set.items(), key=lambda kv: -kv[1])),
            "mean_z": float(zs.mean()) if len(zs) else None,
            "examples": [r["name"] for r in sorted(hits, key=lambda r: -r["z"])[:5]],
        }
    return out


if __name__ == "__main__":
    setcode = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    res = scan(setcode)
    json.dump(res, open(f"{ROOT}/data/precedent_{setcode}.json", "w"), indent=1)
    print(f"{setcode} mechanics vs. 21 training sets\n")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["n_precedent"]):
        mz = f"{v['mean_z']:+.2f}" if v["mean_z"] is not None else "  n/a"
        print(f"  {k:22s} {v['n_precedent']:5d} precedents  mean z {mz}")
        if v["sets"]:
            top = list(v["sets"].items())[:5]
            print(f"      {', '.join(f'{s}:{n}' for s, n in top)}")
        if v["examples"]:
            print(f"      best: {', '.join(v['examples'][:3])}")
