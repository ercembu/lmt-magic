"""Index recorded games for the viewer (sim/viewer.html).

    sim_index.py <recdir> [card_values.json] [SET]

Reads <recdir>/games/*.json.gz (written by sim/SimRecorder.java) and writes
<recdir>/index.json -- one row per game: who won, how long it took, what got
cast, and, given fitted card values from sim_fit.py, how strong each deck was,
so upsets can be found -- plus <recdir>/cards.json (name -> Scryfall image) and
a copy of the viewer. Serve the folder with `python3 -m http.server`.
"""
import glob, gzip, json, os, re, shutil, sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAG = re.compile(r"<[^>]+>")


def image_map(setcode):
    out = {}
    for c in json.load(open(f"{ROOT}/data/cards/{setcode}.json")):
        uris = c.get("image_uris") or (c.get("card_faces") or [{}])[0].get("image_uris") or {}
        if not uris:
            continue
        for n in {c["name"], c["name"].split(" // ")[0]}:
            out.setdefault(n, {"small": uris.get("small"), "normal": uris.get("normal")})
    return out


def main(recdir, values_path=None, setcode="FRA"):
    vals = {}
    if values_path and os.path.exists(values_path):
        vals = {k: v[0] for k, v in json.load(open(values_path)).items()}
    manifest = {}
    mp = glob.glob(os.path.join(recdir, "..", "simdecks", "manifest.json"))
    if mp:
        manifest = {m["deck"]: m for m in json.load(open(mp[0]))}
    labels = {}
    lp = os.path.join(recdir, "labels.json")      # self-play: who played each seat (selfplay_stats.py)
    if os.path.exists(lp):
        labels = json.load(open(lp))
    rows = []
    for f in sorted(glob.glob(os.path.join(recdir, "games", "*.json.gz"))):
        try:
            g = json.load(gzip.open(f, "rt", encoding="utf8"))
        except (OSError, EOFError, ValueError):
            continue        # moved, or still being written
        cast = {"A": [], "B": []}
        for s in g["snapshots"]:
            for line in s["log"]:
                m = re.match(r"^(?:Player)?(A|B) casts (.+?)(?: \[[0-9a-f]+\])?(?: from| targeting|$)", TAG.sub("", line))
                if m:
                    cast[m.group(1)].append(m.group(2))
        gen = re.match(r"gen(\d+)", os.path.basename(f))
        row = {"file": "games/" + os.path.basename(f), "deckA": g["deckA"], "deckB": g["deckB"],
               "gen": int(gen.group(1)) if gen else None, "agent": g.get("agent", "stock"),
               "winner": g["winner"], "turns": g["turns"], "steps": len(g["snapshots"]),
               "castA": cast["A"], "castB": cast["B"]}
        lab = labels.get(os.path.basename(f).split("_G")[0])
        if lab:
            row.update(seatA=lab["A"], seatB=lab["B"], kind=lab["kind"])
        for side, deck in (("A", g["deckA"][:-4]), ("B", g["deckB"][:-4])):
            m = manifest.get(deck, {})
            row["pair" + side] = m.get("pair", "")
            if vals and m:
                row["str" + side] = round(sum(vals.get(c, 0.0) for c in m["cards"]), 3)
        if "strA" in row and g["winner"] in ("A", "B"):
            # positive = the deck the card values rated weaker won
            gap = row["strA"] - row["strB"]
            row["upset"] = round(-gap if g["winner"] == "A" else gap, 3)
        rows.append(row)
    json.dump(rows, open(os.path.join(recdir, "index.json"), "w"))
    json.dump(image_map(setcode), open(os.path.join(recdir, "cards.json"), "w"))
    shutil.copy(os.path.join(ROOT, "sim", "viewer.html"), os.path.join(recdir, "index.html"))
    n_up = sum(1 for r in rows if r.get("upset", 0) > 0)
    print(f"{len(rows)} games indexed"
          + (f", {n_up} won by the weaker deck" if vals else " (no card values yet: no upset ranking)"))
    print(f"serve with:  cd {recdir} && python3 -m http.server 8742   ->  http://localhost:8742/")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
         sys.argv[3] if len(sys.argv) > 3 else "FRA")
