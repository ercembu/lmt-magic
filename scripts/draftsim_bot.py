"""Draftsim's draft/sealed simulator tables: per set, each card's bot rating and archetype tags.

    draftsim_bot.py            fetch the live app and write data/draftsim_bot/<SET>.json

The simulator (draftsim.com/draft/) ships one tab-separated table per set inside its
JavaScript bundle: Name, Casting_Cost, Card_Type, Rarity, Rating, Archetype, Fixing,
Splashable. Its sealed-pool scores are sums of the Rating column.

What we use: the Archetype column, e.g. "[Land Diversity (UG) - Enabler]". It says which
two-colour deck a card serves and whether it enables or pays off, which is exactly what the
model cannot see (it reads each card alone). signals.py shows these per pair.

What we do NOT use: the Rating column. Archived copies of the app (web.archive.org,
Apr-Oct 2026) show three stages per set:
  - early placeholders: SOS 0.378, MSH 0.283 spearman with 17lands GIH WR (the Draftsim
    review: 0.540, 0.437);
  - days before release, a copy of the set review (HOB, FRA: 0.99 agreement with it);
  - after release, a rewrite from results (0.96-0.99 with 17lands).
Mean over SOS/MSH/HOB/FRA, last pre-release copy: bot 0.447, review 0.516, blend 0.507.
So pre-release it is worse than the review and adds nothing to it, and on past sets its
current values are contaminated by the answers.
"""
import json, os, re, sys, urllib.request

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "draftsim_bot")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140.0 Safari/537.36"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read().decode(
        "utf8", "replace")


def tables(bundle):
    """{set code: [row dicts]} from one app bundle."""
    out = {}
    for code, var in re.findall(r'"\.\./data/([A-Za-z0-9_]+)\.txt":(\w+)', bundle):
        m = re.search(r"\b" + re.escape(var) + r"\s*=\s*`(Name\t.*?)`", bundle, re.S)
        if not m:
            continue
        rows = [line.rstrip("\r").split("\t") for line in m.group(1).split("\n")]
        out[code] = [dict(zip(rows[0], r)) for r in rows[1:] if len(r) > 3]
    return out


def archetypes(row):
    """'[Land Diversity (UG) - Enabler][Landfall Value (RG) - Payoff]' ->
    [('UG', 'Land Diversity', 'Enabler'), ('RG', 'Landfall Value', 'Payoff')]"""
    out = []
    for name, pair, role in re.findall(r"\[?\s*([^\[\]()]+?)\s*\((\w{2})\)\s*-\s*(\w+)\s*\]?",
                                       row.get("Archetype", "")):
        out.append((pair.upper(), name.strip(), role.strip().capitalize()))
    return out


if __name__ == "__main__":
    page = get("https://draftsim.com/draft/")
    src = re.search(r'src="(/draft-app/dist/assets/index-[^"]+\.js)"', page)
    if not src:
        raise SystemExit("no app bundle found on draftsim.com/draft/")
    tabs = tables(get("https://draftsim.com" + src.group(1)))
    os.makedirs(OUT, exist_ok=True)
    for code, rows in tabs.items():
        json.dump(rows, open(f"{OUT}/{code}.json", "w"))
    print(f"{len(tabs)} set tables -> {OUT}: {' '.join(sorted(tabs))}")
