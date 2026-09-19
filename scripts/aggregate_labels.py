"""Compute per-card performance metrics from 17lands public game data.

The free 17lands aggregate API is patron-gated (dates ignored, win rates stripped
for anything but the current set), so we derive the numbers ourselves from the
public game dumps. One row per game; per card there are five columns:
opening_hand_X, drawn_X, tutored_X, deck_X, sideboard_X, all copy counts.

Metrics (copy-weighted, matching 17lands' own definitions):
  GIH WR  - win rate of games where a copy was in hand (opening hand/drawn/tutored).
            The standard "how good is this card" number.
  GND WR  - win rate of games where a copy sat in the deck and was never drawn.
            Proxy for the strength of decks that play the card.
  IWD     - GIH WR minus GND WR. Isolates the card's own contribution from the
            quality of the decks it ends up in.
"""
import csv, gzip, os, sys, json, time
import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "labels")
TOP_RANKS = {"platinum", "diamond", "mythic"}
CHUNK = 40000


def card_columns(path):
    """Read just the header and group card columns by card name."""
    with gzip.open(path, "rt", encoding="utf8") as f:
        header = next(csv.reader(f))
    cards, meta = {}, {}
    for i, col in enumerate(header):
        for pref in ("opening_hand_", "drawn_", "tutored_", "deck_", "sideboard_"):
            if col.startswith(pref):
                cards.setdefault(col[len(pref):], {})[pref[:-1]] = col
                break
        else:
            meta[col] = i
    return header, cards, meta


def aggregate(expansion, fmt="PremierDraft"):
    path = os.path.join(RAW, f"game_data_public.{expansion}.{fmt}.csv.gz")
    header, cards, meta = card_columns(path)
    names = sorted(cards)
    # Sets before ~2022 have no tutored_ columns at all. Those are required only
    # where present; dropping the set over it would cost a third of the training
    # data, and tutored copies are a rounding error in GIH either way.
    names = [n for n in names if all(k in cards[n] for k in
                                     ("opening_hand", "drawn", "deck"))]
    has_tu = all("tutored" in cards[n] for n in names) if names else False
    oh = [cards[n]["opening_hand"] for n in names]
    dr = [cards[n]["drawn"] for n in names]
    tu = [cards[n]["tutored"] for n in names] if has_tu else []
    dk = [cards[n]["deck"] for n in names]
    usecols = ["won", "rank"] + oh + dr + tu + dk

    nc = len(names)
    acc = {g: {k: np.zeros(nc, dtype=np.int64) for k in
               ("gih_n", "gih_w", "gnd_n", "gnd_w", "deck_n", "deck_w")}
           for g in ("all", "top")}
    games = {"all": 0, "top": 0}
    wins = {"all": 0, "top": 0}

    t0, rows = time.time(), 0
    reader = pd.read_csv(path, usecols=usecols, chunksize=CHUNK,
                         compression="gzip", low_memory=False)
    for chunk in reader:
        rows += len(chunk)
        won = (chunk["won"].astype(str).values == "True")
        rank = chunk["rank"].astype(str).str.lower().values
        gih = (chunk[oh].to_numpy(dtype=np.int32, na_value=0)
               + chunk[dr].to_numpy(dtype=np.int32, na_value=0))
        if has_tu:
            gih = gih + chunk[tu].to_numpy(dtype=np.int32, na_value=0)
        deck = chunk[dk].to_numpy(dtype=np.int32, na_value=0)
        gnd = np.maximum(deck - gih, 0)

        for group in ("all", "top"):
            m = np.ones(len(chunk), bool) if group == "all" else np.isin(rank, list(TOP_RANKS))
            if not m.any():
                continue
            w = won[m]
            a = acc[group]
            g_, d_, n_ = gih[m], deck[m], gnd[m]
            a["gih_n"] += g_.sum(0);            a["gih_w"] += (g_ * w[:, None]).sum(0)
            a["gnd_n"] += n_.sum(0);            a["gnd_w"] += (n_ * w[:, None]).sum(0)
            a["deck_n"] += d_.sum(0);           a["deck_w"] += (d_ * w[:, None]).sum(0)
            games[group] += int(m.sum());       wins[group] += int(w.sum())
        if rows % (CHUNK * 10) == 0:
            print(f"  {expansion}: {rows:,} rows  ({time.time()-t0:.0f}s)", flush=True)

    out = {"expansion": expansion, "format": fmt, "rows": rows,
           "baseline": {g: (wins[g] / games[g] if games[g] else None) for g in games},
           "games": games, "cards": {}}
    for group in ("all", "top"):
        a = acc[group]
        for i, n in enumerate(names):
            rec = out["cards"].setdefault(n, {})
            gih_n, gnd_n, deck_n = int(a["gih_n"][i]), int(a["gnd_n"][i]), int(a["deck_n"][i])
            rec[group] = {
                "gih_n": gih_n, "gih_wr": (a["gih_w"][i] / gih_n) if gih_n else None,
                "gnd_n": gnd_n, "gnd_wr": (a["gnd_w"][i] / gnd_n) if gnd_n else None,
                "deck_n": deck_n, "deck_wr": (a["deck_w"][i] / deck_n) if deck_n else None,
            }
            g, d = rec[group]["gih_wr"], rec[group]["gnd_wr"]
            rec[group]["iwd"] = (g - d) if (g is not None and d is not None) else None
    print(f"{expansion}: {rows:,} games, {nc} cards, baseline "
          f"{out['baseline']['all']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fmt = os.environ.get("FORMAT", "PremierDraft")
    suffix = "" if fmt == "PremierDraft" else f".{fmt}"
    for exp in sys.argv[1:]:
        dst = os.path.join(OUT, f"{exp}{suffix}.json")
        if os.path.exists(dst):
            print(f"{exp}: cached"); continue
        try:
            res = aggregate(exp, fmt)
        except Exception as e:
            print(f"{exp}: FAILED {type(e).__name__}: {e}"); continue
        with open(dst, "w") as f:
            json.dump(res, f)
