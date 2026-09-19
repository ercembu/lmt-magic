"""Download and parse WotC prerelease guides for every set we can get one for.

Two traps this guards against, both of which bit us:

  * magic.wizards.com serves soft-404s -- HTTP 200 with a "PAGE NOT FOUND" body --
    and redirects some misses to the Daily MTG index, which embeds whatever guide
    is currently featured. Scraping that silently produced a 'MKM' setinfo file
    full of Reality Fracture cards.
  * Guides change layout between sets, so a parse that "succeeds" can still be
    nonsense.

So: verify the page is the guide we asked for, then verify the parsed card names
are actually cards in that set. Anything else is rejected rather than written.
"""
import json, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
import parse_guide

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "guides")

# Real slugs, taken from the news archive rather than guessed -- suffixes vary.
SLUGS = {
    "FRA": "reality-fracture-prerelease-guide",
    "HOB": "the-hobbit-prerelease-guide",
    "MSH": "marvel-super-heroes-prerelease-guide",
    "SOS": "secrets-of-strixhaven-prerelease-guide",
    "TMT": "teenage-mutant-ninja-turtles-prerelease-guide",
    "ECL": "lorwyn-eclipsed-prerelease-guide",
    "TLA": "avatar-the-last-airbender-prerelease-guide",
    "EOE": "edge-of-eternities-prerelease-guide",
    "FIN": "final-fantasy-prerelease-guide",
    "TDM": "tarkir-dragonstorm-prerelease-guide",
    "DFT": "the-guide-to-your-first-prerelease-with-aetherdrift",
    "DSK": "duskmourn-house-of-horror-prerelease-and-draft-guide",
    "BLB": "bloomburrow-prerelease-guide",
    "OTJ": "outlaws-of-thunder-junction-prerelease-primer",
    "FDN": "foundations-prerelease-guide",
    # No prerelease guide was published for MKM, LCI, WOE, MOM, ONE, BRO, DMU, LTR.
}
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140.0 Safari/537.36"


def fetch(slug, dst):
    url = f"https://magic.wizards.com/en/news/feature/{slug}"
    subprocess.run(["curl", "-s", "-m", "60", "-L", "-H", f"User-Agent: {UA}",
                    "-o", dst, url], check=True)
    return open(dst, encoding="utf8", errors="replace").read()


def is_real_guide(html, slug):
    """Reject soft-404s and redirects to the news index."""
    title = re.search(r"<title>([^<]*)", html)
    title = (title.group(1) if title else "").lower()
    if "404" in title or "page not found" in html.lower()[:400000] and "PAGE NOT FOUND" in html:
        return False, f"soft 404 (title: {title[:40]})"
    if "daily mtg" in title and "prerelease" not in title:
        return False, "redirected to the news index"
    # the slug's own words should appear in the title
    words = [w for w in slug.split("-") if len(w) > 3
             and w not in ("prerelease", "guide", "primer", "with", "your", "first", "draft")]
    if words and not any(w in title for w in words):
        return False, f"title doesn't match slug (title: {title[:40]})"
    return True, "ok"


def validate(info, setcode):
    """Parsed names must be cards in this set."""
    path = f"{ROOT}/data/cards/{setcode}.json"
    if not os.path.exists(path):
        return False, "no card data for this set"
    names = {c["name"] for c in json.load(open(path))}

    def res(n):
        return n in names or any(k.split(" // ")[0] == n or k.split(",")[0] == n for k in names)

    all_named = info["removal"] + [n for a in info["archetypes"].values() for n in a["key_cards"]]
    if not all_named:
        return False, "parsed no card names"
    ok = sum(1 for n in all_named if res(n))
    frac = ok / len(all_named)
    if frac < 0.8:
        return False, f"only {ok}/{len(all_named)} named cards belong to {setcode}"
    return True, f"{ok}/{len(all_named)} names check out"


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(f"{ROOT}/data/setinfo", exist_ok=True)
    want = sys.argv[1:] or list(SLUGS)
    good = []
    for code in want:
        slug = SLUGS.get(code)
        if not slug:
            print(f"{code:5s} no guide published"); continue
        dst = os.path.join(CACHE, f"{code}.html")
        if not os.path.exists(dst):
            fetch(slug, dst)
        html = open(dst, encoding="utf8", errors="replace").read()
        real, why = is_real_guide(html, slug)
        if not real:
            print(f"{code:5s} REJECTED page — {why}"); os.remove(dst); continue
        try:
            info = parse_guide.parse(dst, code)
        except SystemExit as e:
            print(f"{code:5s} unparsed — {e}"); continue
        ok, why = validate(info, code)
        if not ok:
            print(f"{code:5s} REJECTED parse — {why}"); continue
        json.dump(info, open(f"{ROOT}/data/setinfo/{code}.json", "w"), indent=1)
        na = sum(1 for a in info["archetypes"].values() if a["key_cards"])
        print(f"{code:5s} ok — {len(info['archetypes'])} archetypes "
              f"({na} with key cards), {len(info['removal'])} removal; {why}")
        good.append(code)
    print(f"\nusable setinfo for {len(good)} sets: {' '.join(good)}")
