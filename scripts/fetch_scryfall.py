"""Fetch full card data from Scryfall for every set in the training pool + the target set."""
import json, time, os, sys, urllib.request, urllib.parse

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "cards")

# 17lands expansion code -> scryfall set code
SETS = {
    "FRA": "fra", "HOB": "hob", "MSH": "msh", "SOS": "sos", "TMT": "tmt",
    "ECL": "ecl", "TLA": "tla", "EOE": "eoe", "FIN": "fin", "TDM": "tdm",
    "DFT": "dft", "DSK": "dsk", "BLB": "blb", "OTJ": "otj", "MKM": "mkm",
    "LCI": "lci", "WOE": "woe", "LTR": "ltr", "MOM": "mom", "ONE": "one",
    "BRO": "bro", "DMU": "dmu",
    # older sets, added to test whether more training data helps
    "FDN": "fdn", "MH3": "mh3", "SIR": "sir", "SNC": "snc", "NEO": "neo",
    "VOW": "vow", "MID": "mid", "AFR": "afr", "STX": "stx", "KHM": "khm",
    "ZNR": "znr", "M21": "m21", "IKO": "iko", "THB": "thb", "ELD": "eld",
}

def get(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": "lmt-magic/0.1 (limited card grader research)",
        "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def fetch_set(code):
    """All unique cards in a set, in the booster-relevant printing."""
    cards, url = [], ("https://api.scryfall.com/cards/search?q=" +
        urllib.parse.quote(f"set:{code}") + "&unique=cards&order=set")
    while url:
        d = get(url)
        cards.extend(d["data"])
        url = d.get("next_page")
        time.sleep(0.12)   # scryfall asks for 50-100ms between requests
    return cards

if __name__ == "__main__":
    want = sys.argv[1:] or list(SETS)
    for exp in want:
        sf = SETS[exp]
        path = os.path.join(OUT, f"{exp}.json")
        if os.path.exists(path):
            print(f"{exp}: cached"); continue
        try:
            cards = fetch_set(sf)
        except Exception as e:
            print(f"{exp}: FAILED {e}"); continue
        with open(path, "w") as f:
            json.dump(cards, f)
        boosterable = sum(1 for c in cards if c.get("booster"))
        print(f"{exp}: {len(cards)} cards ({boosterable} in boosters)")
