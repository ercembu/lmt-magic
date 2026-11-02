"""Live progress of a simulation run, for the viewer's Progress tab.

    sim_stats.py <results.tsv> <pairs.tsv> <manifest.json> <runner.log> <out.json> [SET]

Games done and rate, win rate by colour pair and by colour, game length, and the
number the whole run exists for: how well the simulated card values currently
agree with the set's real 17lands win rates.
"""
import json, math, os, re, sys, time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import sim_fit


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


# The post-mortem's numbers for the frozen pre-release model (postmortem.py),
# so the curve has something to be measured against.
REFS = {"FRA": [{"label": "our model, blind (Sealed)", "fmt": "Sealed", "rho": 0.516},
                {"label": "our model, with reviews (Sealed)", "fmt": "Sealed", "rho": 0.613}]}
STEP = 500


def learning_curve(results, manifest, setcode, rate):
    """Accuracy after the first 500, 1000, ... finished games, in completion order.

    Each checkpoint is fitted once and cached next to the results, so this stays
    cheap however long the run gets. The clock time of a checkpoint is recorded
    when it is first computed; ones computed in a backlog are estimated from the
    current rate and marked as such.
    """
    from scipy.stats import spearmanr
    cache_p = os.path.join(os.path.dirname(results), "curve.json")
    curve = json.load(open(cache_p)) if os.path.exists(cache_p) else []
    games = sim_fit.load_games(results, manifest)
    have = {c["games"] for c in curve}
    truth = {f: sim_fit.truth(setcode, f) for f in ("Sealed", "PremierDraft")}
    new = [n for n in range(STEP, len(games) + 1, STEP) if n not in have]
    for n in new:
        vals = sim_fit.fit(games[:n])
        row = {"games": n, "t": time.strftime("%H:%M"), "estimated_time": n != new[-1]}
        for f, t in truth.items():
            ks = [k for k, (v, c) in vals.items() if k in t and c >= 10]
            row[f] = round(float(spearmanr([vals[k][0] for k in ks], [t[k] for k in ks]).statistic), 3)
        curve.append(row)
    curve.sort(key=lambda c: c["games"])
    if new:
        json.dump(curve, open(cache_p, "w"))
    return curve


def main(results, pairs, manifest, log, out, setcode="FRA"):
    decks = {m["deck"]: m for m in json.load(open(manifest))}
    total = sum(1 for _ in open(pairs))
    rows = [l.rstrip("\n").split("\t") for l in open(results)] if os.path.exists(results) else []
    res = Counter(r[2] for r in rows if len(r) > 2)
    rate = None
    if os.path.exists(log):
        # XMage's own logging floods this file, so read a generous tail
        with open(log, "rb") as f:
            f.seek(max(0, os.path.getsize(log) - 2_000_000))
            m = re.findall(r"([\d.]+) games/min", f.read().decode("utf8", "replace"))
        rate = float(m[-1]) if m else None
    done = sum(v for k, v in res.items() if k != "error")
    eta = (total - done) / rate if rate else None

    pair = defaultdict(lambda: [0, 0])
    colour = defaultdict(lambda: [0, 0])
    turns = Counter()
    for r in rows:
        if len(r) < 4 or r[2] not in ("A", "B"):
            continue
        a, b = (os.path.basename(x)[:-4] for x in r[:2])
        pa, pb = decks[a]["pair"], decks[b]["pair"]
        turns[min(int(r[3]), 30)] += 1
        if pa == pb:
            continue                             # a mirror says nothing about the pair
        win, lose = (pa, pb) if r[2] == "A" else (pb, pa)
        pair[win][0] += 1; pair[win][1] += 1; pair[lose][1] += 1
        for c in set(win) - set(lose):
            colour[c][0] += 1; colour[c][1] += 1
        for c in set(lose) - set(win):
            colour[c][1] += 1

    stats = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "set": setcode, "total": total, "done": done, "errors": res.get("error", 0),
        "draws": res.get("draw", 0), "rate": rate, "eta_min": round(eta) if eta else None,
        "pairs": sorted([{"pair": p, "wins": w, "games": n, "wr": round(w / n, 3), "ci": wilson(w, n)}
                         for p, (w, n) in pair.items()], key=lambda x: -x["wr"]),
        "colours": sorted([{"colour": c, "wins": w, "games": n, "wr": round(w / n, 3), "ci": wilson(w, n)}
                           for c, (w, n) in colour.items()], key=lambda x: -x["wr"]),
        "turns": {str(k): v for k, v in sorted(turns.items())},
    }

    games = sim_fit.load_games(results, manifest) if rows else []
    if len(games) >= 200:
        from scipy.stats import spearmanr
        vals = sim_fit.fit(games)
        acc = {}
        for fmt in ("Sealed", "PremierDraft"):
            t = sim_fit.truth(setcode, fmt)
            ks = [k for k, (v, n) in vals.items() if k in t and n >= 30]
            if len(ks) >= 20:
                acc[fmt] = {"rho": round(float(spearmanr([vals[k][0] for k in ks],
                                                         [t[k] for k in ks]).statistic), 3),
                            "cards": len(ks)}
        ranked = sorted(((v, n, k) for k, (v, n) in vals.items() if n >= 30), reverse=True)
        stats["accuracy"] = acc
        stats["top"] = [{"card": k, "value": round(v, 3), "decks": n} for v, n, k in ranked[:12]]
        stats["bottom"] = [{"card": k, "value": round(v, 3), "decks": n} for v, n, k in ranked[-12:][::-1]]
        json.dump(vals, open(os.path.join(os.path.dirname(results), "card_values.json"), "w"))
    stats["curve"] = learning_curve(results, manifest, setcode, rate)
    # what the run is trying to beat: our model's own score on FRA, blind and with reviews
    stats["refs"] = REFS.get(setcode, [])
    tmp = out + ".tmp"
    json.dump(stats, open(tmp, "w"))
    os.replace(tmp, out)


if __name__ == "__main__":
    main(*sys.argv[1:6], *(sys.argv[6:7] or ["FRA"]))
