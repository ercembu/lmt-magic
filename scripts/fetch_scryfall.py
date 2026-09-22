"""Fetch full card data from Scryfall for every set in the training pool + the target set."""
import json, time, os, sys, urllib.request, urllib.parse, urllib.error

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

def search(query):
    cards, url = [], ("https://api.scryfall.com/cards/search?q=" +
        urllib.parse.quote(query) + "&unique=cards&order=set")
    while url:
        try:
            d = get(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:        # scryfall 404s an empty result set
                return cards
            raise
        cards.extend(d["data"])
        url = d.get("next_page")
        time.sleep(0.12)   # scryfall asks for 50-100ms between requests
    return cards


_SETS_INDEX = None
_SPG = None

def special_guests():
    """The standing Special Guests set, fetched once for the whole run."""
    global _SPG
    if _SPG is None:
        _SPG = search("set:spg")
    return _SPG


def sets_index():
    global _SETS_INDEX
    if _SETS_INDEX is None:
        _SETS_INDEX = get("https://api.scryfall.com/sets")["data"]
    return _SETS_INDEX


# A set's Play Boosters hold more than the set. Two kinds of bonus sheet turn up
# in the draft environment and are openable at a prerelease:
#
#   a child set     Breaking News (otp), The Big Score (big), Avatar Eternal (tle),\n#                   Mystical Archive
#                   (soa), Stellar Sights (eos), Through the Ages (fca)
#   Special Guests  a standing set (spg) whose cards are dealt out to whichever
#                   release they ship with, identifiable by sharing its date
#
# Tokens, promos, art series, commander decks and Alchemy are not in boosters.
# "eternal" children (Avatar Eternal, TMNT Eternal) were tried and rejected:
# they recover 45 labelled TLA cards but drag in 242 that are not in the draft
# environment at all, and anything computed per set -- colour depth, the grade
# curve -- would then be measured over cards nobody can open.
BONUS_CHILD_TYPES = {"masterpiece", "expansion"}

def bonus_codes(code):
    """(child set codes, parent release date) for one parent set."""
    idx = sets_index()
    parent = next((s for s in idx if s["code"] == code), None)
    if parent is None:
        return [], None
    kids = [s["code"] for s in idx
            if s.get("parent_set_code") == code
            and s["set_type"] in BONUS_CHILD_TYPES]
    return kids, parent.get("released_at")


def fetch_set(code):
    """Everything openable in this set's boosters, bonus sheets included."""
    cards = search(f"set:{code}")
    have = {c["id"] for c in cards}
    kids, released = bonus_codes(code)
    for kid in kids:
        for c in search(f"set:{kid}"):
            if c["id"] not in have:
                have.add(c["id"])
                c["bonus_sheet"] = kid
                cards.append(c)
    # Special Guests ship with a release rather than belonging to one, so the
    # shared date is what ties them to this set.
    if released:
        for c in special_guests():
            if c.get("released_at") == released and c["id"] not in have:
                have.add(c["id"])
                c["bonus_sheet"] = "spg"
                cards.append(c)
    return cards

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)      # first run / CI cache miss has no data/cards
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
        bonus = sum(1 for c in cards if c.get("bonus_sheet"))
        print(f"{exp}: {len(cards)} cards"
              + (f" (+{bonus} from bonus sheets)" if bonus else ""))
