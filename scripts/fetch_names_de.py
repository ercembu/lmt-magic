"""German card names from Scryfall for the site's "DE" switch and the card finder.

    fetch_names_de.py SET [SET ...]   -> data/names_de/<SET>.json  {english name: german name}

The set's own German prints first; a card with none (a set not out in German yet) takes
the name of a German print from an older set, so reprints are named. Double-faced cards
also map each face. New cards with no German print at all are left out: the finder falls
back on data/names_de/<SET>.unofficial.json, translations made before release.
"""
import json, os, sys, time, urllib.error, urllib.parse, urllib.request

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "names_de")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "lmt-magic/0.1 (limited card grader research)",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def names(code):
    out = search(f"set:{code.lower()} lang:de")
    grades = os.path.join(os.path.dirname(__file__), "..", "data", f"grades_{code.upper()}.json")
    missing = [c["name"] for c in json.load(open(grades))] if os.path.exists(grades) else []
    missing = [n for n in missing if n not in out]
    for i in range(0, len(missing), 15):   # reprints: any German print, 15 names per query
        q = " or ".join('!"' + n.replace('"', '') + '"' for n in missing[i:i + 15])
        out.update({k: v for k, v in search(f"lang:de ({q})").items() if k not in out})
    return out


def search(q):
    out, url = {}, "https://api.scryfall.com/cards/search?unique=prints&q=" + urllib.parse.quote(q)
    while url:
        try:
            d = get(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:   # no German prints yet
                break
            raise
        for c in d["data"]:
            if c.get("printed_name"):
                out[c["name"]] = c["printed_name"]
            faces = [f for f in c.get("card_faces", []) if f.get("printed_name")]
            for f in faces:
                out[f["name"]] = f["printed_name"]
            if faces and c["name"] not in out:
                out[c["name"]] = " // ".join(f["printed_name"] for f in faces)
        url = d.get("next_page")
        time.sleep(0.12)
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for code in sys.argv[1:]:
        n = names(code)
        json.dump(n, open(os.path.join(OUT, f"{code.upper()}.json"), "w"), ensure_ascii=False, indent=0, sort_keys=True)
        print(code.upper(), len(n), "German names")
