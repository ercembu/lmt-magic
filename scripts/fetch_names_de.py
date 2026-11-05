"""German card names from Scryfall for the site's "DE" switch.

    fetch_names_de.py SET [SET ...]   -> data/names_de/<SET>.json  {english name: german name}

Double-faced cards also map each face. A set Scryfall has no German prints of yet
gets an empty file, and the site keeps the English names.
"""
import json, os, sys, time, urllib.error, urllib.parse, urllib.request

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "names_de")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "lmt-magic/0.1 (limited card grader research)",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def names(code):
    out, url = {}, ("https://api.scryfall.com/cards/search?unique=prints&q="
                    + urllib.parse.quote(f"set:{code.lower()} lang:de"))
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
