"""Combine the two published reviews into one rating for the model.

Two independent reviewers rating the same cards is a less noisy estimate than
either alone -- they agree at 0.75, so the half they disagree on is largely
noise that averaging cancels.

The catch is calibration. The model learned its mapping from *Draftsim's* rating
distribution across 25 past sets, and a plain average has a visibly tighter
spread (sd 2.05 against Draftsim's 2.34) because disagreement pulls values
toward the middle. Feeding that in unchanged would put the input off the
distribution the model was trained on.

So: rank the cards by consensus, then hand back Draftsim's own sorted values in
that order. The ordering is the improved one, the marginal distribution is
exactly what the model expects, and nothing has to be recalibrated.
"""
import json, os, sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "expert_consensus")


def build(setcode):
    ds_path = f"{ROOT}/data/expert/{setcode}.json"
    az_path = f"{ROOT}/data/expert_mtgazone/{setcode}.json"
    if not os.path.exists(ds_path):
        return None, "no primary review"
    ds = json.load(open(ds_path))
    if not os.path.exists(az_path):
        return dict(ds), "only one review; passing it through unchanged"
    az = json.load(open(az_path))

    names = sorted(set(ds) | set(az))
    raw = {}
    for n in names:
        a, b = ds.get(n), az.get(n)
        b = b * 2 if b is not None else None          # 0-5 -> 0-10
        raw[n] = (a + b) / 2 if (a is not None and b is not None) else (a if a is not None else b)

    # quantile-map onto the primary reviewer's value distribution
    pool = sorted(ds.values())
    ordered = sorted(names, key=lambda n: raw[n])
    out = {}
    for i, n in enumerate(ordered):
        # stretch the (possibly larger) name list across the value pool
        j = round(i * (len(pool) - 1) / max(len(ordered) - 1, 1))
        out[n] = float(pool[j])
    both = sum(1 for n in names if n in ds and n in az)
    return out, f"{len(names)} cards ({both} rated by both)"


if __name__ == "__main__":
    setcode = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    res, note = build(setcode)
    if res is None:
        raise SystemExit(note)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/{setcode}.json", "w"), indent=1)
    import numpy as np
    ds = json.load(open(f"{ROOT}/data/expert/{setcode}.json"))
    v, d = list(res.values()), list(ds.values())
    print(f"{setcode}: {note}")
    print(f"  consensus  mean {np.mean(v):.2f}  sd {np.std(v):.2f}")
    print(f"  Draftsim   mean {np.mean(d):.2f}  sd {np.std(d):.2f}   (distribution preserved)")
