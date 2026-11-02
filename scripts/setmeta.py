"""Write set.json: which set a build grades, for the pages to name it.

The pages used to say "Reality Fracture" and "280 cards" in their markup, which
meant moving on to the next set was an edit to four HTML files. They now mark
those words with data-set="..." and pwa.js fills them in from this file.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import train

ROOT = os.path.join(os.path.dirname(__file__), "..")


def build(setcode):
    cards = json.load(open(f"{ROOT}/data/cards/{setcode}.json"))
    name = next(c["set_name"] for c in cards if not c.get("bonus_sheet"))
    grades = json.load(open(f"{ROOT}/data/grades_{setcode}.json"))
    trained = [s for s in train.TRAIN_SETS if s != setcode
               and os.path.exists(f"{ROOT}/data/labels/{s}.json")]
    return {"code": setcode, "name": name, "cards": len(grades),
            "train_sets": len(trained)}


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "TRK"
    out = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/data/set_{code}.json"
    meta = build(code)
    json.dump(meta, open(out, "w"))
    print(f"{out}: {meta}")
