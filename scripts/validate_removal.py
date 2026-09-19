"""Check the regex removal detector against WotC's published removal list.

WotC's list is 'key removal', so our detector legitimately flags more cards than
theirs (fringe/situational removal). What matters is recall: every card WotC
calls removal should trip the detector.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import features

ROOT = os.path.join(os.path.dirname(__file__), "..")


def resolve(name, cards):
    if name in cards:
        return name
    return next((k for k in cards if k.split(" // ")[0] == name
                 or k.split(",")[0] == name), None)


def check(setcode):
    cards = {c["name"]: c for c in json.load(open(f"{ROOT}/data/cards/{setcode}.json"))}
    info = json.load(open(f"{ROOT}/data/setinfo/{setcode}.json"))
    official = {resolve(n, cards) for n in info["removal"]} - {None}
    flagged = {n for n, c in cards.items() if features.extract(c)["e_removal_any"]}
    hit = official & flagged
    missed = official - flagged
    print(f"{setcode}: WotC lists {len(official)} removal spells; "
          f"detector flags {len(flagged)} cards set-wide")
    print(f"  recall {len(hit)}/{len(official)} = {len(hit)/len(official):.0%}")
    if missed:
        print("  MISSED:")
        for n in sorted(missed):
            ot = (cards[n].get("oracle_text") or "").replace("\n", " / ")
            print(f"    {n:32s} {ot[:100]}")
    return len(hit), len(official)


if __name__ == "__main__":
    check(sys.argv[1] if len(sys.argv) > 1 else "FRA")
