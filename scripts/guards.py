"""Set-specific guard checks layered on top of the statistical model.

The model is deliberately set-agnostic: it only knows features that exist in
every set, which is what lets it grade an unreleased one. The cost is that it is
blind to a set's own mechanics and archetypes. This module supplies that
knowledge from WotC's published prerelease material and applies it as small,
bounded, *explained* adjustments -- never a silent fudge.

Two jobs:
  1. Adjust  - a card that is a defined archetype payoff plays above its raw
               stats in that archetype; reflect that, modestly.
  2. Flag    - a card leaning on a mechanic with no analogue in the training
               sets is a card the model is guessing about. Say so rather than
               pretending the number is solid.
"""
import json, os, re

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Mechanics unique to a set, with the closest precedent in the training pool.
# 'precedent' None means the model has genuinely never seen anything like it.
# Mechanics unique to a set.
#   pattern  - matches the card in THIS set
#   analogue - matches mechanically equivalent cards in older sets, used to
#              measure how much precedent the model actually has (precedent_scan.py).
#              Literal token names never appear in older sets, so matching on the
#              effect rather than the name is what makes the count meaningful.
SET_MECHANICS = {
    "FRA": {
        "empower": {
            "pattern": r"empower (jace|liliana|chandra|garruk|ajani)",
            "analogue": r"loyalty counter|planeswalker token",
            "desc": "Empower — puts loyalty counters on a planeswalker token, creating one if needed",
            "archetypes": ["UG", "WU"],
        },
        "threshold": {
            "pattern": r"threshold|seven or more cards in your graveyard",
            "analogue": r"seven or more cards in your graveyard|delirium|four or more card types",
            "desc": "Threshold — switches on at 7+ cards in your graveyard",
            "archetypes": ["UB"],
        },
        "behold": {
            "pattern": r"\bbehold\b",
            "analogue": r"\bbehold\b|as an additional cost to cast this spell, reveal",
            "desc": "Behold — cost reduction for revealing/controlling a qualifying permanent",
            "archetypes": ["UG"],
        },
        "heartwood": {
            "pattern": r"heartwood",
            "analogue": r"treasure token|token.{0,60}add \{|\bmana\b.{0,30}token",
            "desc": "Heartwood token — a token that ramps you",
            "archetypes": ["RG"],
        },
        "cadet": {
            "pattern": r"cadet",
            "analogue": r"create (a|two|three|four|x|\w+) \d+/\d+ .{0,60}creature token",
            "desc": "Cadet token — 2/2 bodies for going wide",
            "archetypes": ["WR"],
        },
        "prepare": {
            "pattern": r"enters prepared|while it's prepared|prepared spell|becomes prepared",
            "analogue": r"enters prepared|while it's prepared|adventure",
            "desc": "Prepare — creature that also casts a copy of its spell half",
            "archetypes": [],
        },
        "planeswalker_matters": {
            "pattern": r"\bplaneswalkers?\b|loyalty counter",
            "analogue": r"\bplaneswalkers?\b|loyalty counter",
            "desc": "Planeswalker-matters — live here because Empower makes Jace tokens",
            "archetypes": ["UG", "WU"],
        },
    },
}

# Mechanical signals that a card serves a given archetype hook. Deliberately
# written in generic rules language so the same detector works on any set.
ARCHETYPE_SIGNALS = {
    "WU": [r"\bsurveil\b", r"\bscry\b", r"look at the top"],
    "UB": [r"\bmills?\b", r"threshold", r"cards in your graveyard", r"from your graveyard"],
    "BR": [r"damage to (target player|each opponent|target opponent)", r"loses \d+ life"],
    "RG": [r"heartwood", r"add \{", r"search your library for a .{0,20}land", r"power \d+ or greater"],
    "WG": [r"you gain \d+ life", r"lifelink", r"\+1/\+1 counter"],
    "WB": [r"when .{0,30}dies", r"sacrifice", r"from your graveyard", r"return .{0,30}graveyard"],
    "UR": [r"whenever you cast a?n? ?(noncreature|instant or sorcery)", r"prowess", r"instant or sorcery"],
    "BG": [r"deathtouch", r"trample", r"when .{0,30}enters", r"when .{0,30}dies"],
    "WR": [r"\+1/\+1 counter", r"create .{0,40}token", r"attacks", r"haste"],
    "UG": [r"empower", r"planeswalker", r"loyalty counter", r"behold"],
}


def card_text(card):
    if "card_faces" in card:
        return "\n".join(f.get("oracle_text", "") or "" for f in card["card_faces"]).lower()
    return (card.get("oracle_text") or "").lower()


def load_setinfo(setcode):
    p = f"{ROOT}/data/setinfo/{setcode}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def load_precedent(setcode):
    p = f"{ROOT}/data/precedent_{setcode}.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def mechanics_used(card, setcode):
    t = card_text(card)
    out = []
    for name, m in SET_MECHANICS.get(setcode, {}).items():
        if re.search(m["pattern"], t):
            prec = load_precedent(setcode).get(name, {})
            out.append({"name": name, "desc": m["desc"],
                        "n_precedent": prec.get("n_precedent"),
                        "precedent_mean_z": prec.get("mean_z"),
                        "precedent_sets": list((prec.get("sets") or {}))[:4],
                        "archetypes": m["archetypes"]})
    return out


def archetype_fit(card, setcode):
    """Every archetype the card can actually be played in, and which it helps.

    These are two different questions and conflating them was wrong: a mono-blue
    bomb is castable in all four blue pairs while supporting none of their
    mechanical themes, and listing nothing for it made it invisible to the
    archetype filter. `castable` answers "can this go in that deck"; `supports`
    answers "does it do what that deck is trying to do".
    """
    info = load_setinfo(setcode)
    t = card_text(card)
    # Colour IDENTITY, not mana cost. Cost looks tempting -- Codie is {3} so
    # anyone can cast it -- but it is wrong for the cases that matter: a dual
    # land has no mana cost at all and would land in all ten archetypes, and a
    # prepare card's spell half ({U/B} on a {1}{B} creature) is the reason you
    # play it. Identity is right for 28 of the 29 cards where the two differ;
    # the exception is Codie, whose five-colour ability you cannot use in a
    # two-colour deck anyway -- which is part of why it grades F.
    ci = set(card.get("color_identity") or [])
    defined = list(info["archetypes"]) if info else list(ARCHETYPE_SIGNALS)
    fits = []
    for pair in defined:
        if not ci <= set(pair):
            continue                        # can't cast it there
        pats = ARCHETYPE_SIGNALS.get(pair, [])
        n = sum(1 for p in pats if re.search(p, t))
        entry = info["archetypes"][pair] if info else None
        fits.append({
            "pair": pair,
            "name": entry["name"] if entry else pair,
            "signals": n,
            "key_card": bool(entry and card["name"] in entry["key_cards"]),
            "supports": bool(n),
        })
    return sorted(fits, key=lambda f: (-f["key_card"], -f["signals"], f["pair"]))


# Measured on held-out released sets (guard_ablation.py): across 11 sets with
# validated setinfo the numeric nudges changed spearman by +0.0005 (p=0.54) and
# mean grade error by -0.0015 (p=0.81) -- helped 3, hurt 2, no change on 6. So
# they are computed and shown as context, but do not move the grade by default.
# Flip to True to re-enable, and re-run guard_ablation.py before believing it.
APPLY_ADJUSTMENTS = False


def apply(card, setcode, base_z, apply_adjustments=None):
    """Return (z, adjustments, flags, confidence).

    `adjustments` are advisory unless APPLY_ADJUSTMENTS is on: things a human
    should weigh that the model cannot see. `flags` and `confidence` are the part
    that earns its place -- they say how much precedent the model actually had.
    """
    info = load_setinfo(setcode)
    adj, flags = [], []
    delta = 0.0

    fits = [f for f in archetype_fit(card, setcode) if f["supports"]]
    key_for = [f for f in fits if f["key_card"]]
    if key_for:
        d = 0.25
        delta += d
        adj.append({"reason": f"WotC names it a key card for {key_for[0]['name']}",
                    "delta": d})
    elif len(fits) >= 3:
        d = 0.10
        delta += d
        adj.append({"reason": f"supports {len(fits)} archetypes — flexible", "delta": d})
    elif len(fits) == 1 and fits[0]["signals"] >= 2:
        d = -0.10
        delta += d
        adj.append({"reason": f"narrow — only pulls weight in {fits[0]['name']}", "delta": d})

    if info and card["name"] in info["removal"]:
        d = 0.20
        delta += d
        adj.append({"reason": "on WotC's key removal list", "delta": d})

    # Confidence is set by how much precedent the model actually has for the
    # mechanics this card leans on -- measured, not assumed.
    mechs = mechanics_used(card, setcode)
    THIN = 25
    novel = [m for m in mechs if (m.get("n_precedent") or 0) < THIN]
    for m in mechs:
        flags.append({"mechanic": m["name"], "desc": m["desc"],
                      "n_precedent": m.get("n_precedent"),
                      "precedent_mean_z": m.get("precedent_mean_z"),
                      "precedent_sets": m.get("precedent_sets")})

    # Thin-precedent mechanics: the model has too few examples to have learned
    # them, so nudge toward what the few precedents actually did. Well-precedented
    # mechanics get nothing -- the model already priced those in from training.
    for m in mechs:
        n, mz = m.get("n_precedent"), m.get("precedent_mean_z")
        if n is None or mz is None or n >= THIN or abs(mz) < 0.3:
            continue
        d = 0.15 if mz > 0 else -0.15
        delta += d
        adj.append({"reason": f"{m['name']}: only {n} precedents in training, and they "
                              f"averaged {mz:+.2f}z — model has little signal here",
                    "delta": d})

    delta = max(-0.5, min(0.5, delta))      # guards nudge, they never decide
    if not (APPLY_ADJUSTMENTS if apply_adjustments is None else apply_adjustments):
        delta = 0.0
    if any((m.get("n_precedent") or 0) == 0 for m in mechs):
        confidence = "low"
    elif novel:
        confidence = "medium"
    else:
        confidence = "high"
    return base_z + delta, adj, flags, confidence
