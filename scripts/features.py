"""Turn a Scryfall card into a feature vector.

Everything here must be computable from card data alone, for any set, with no
set-specific knowledge -- that is what lets a model trained on 21 released sets
score an unreleased one. Set-specific mechanics are handled separately, in the
guard layer (guards.py), not here.
"""
import re
import numpy as np

EVERGREEN = ["flying", "trample", "deathtouch", "lifelink", "first strike",
             "double strike", "vigilance", "menace", "haste", "hexproof",
             "ward", "reach", "defender", "flash", "prowess", "indestructible"]


def faces(card):
    """Oracle text of all faces joined; DFCs/split cards keep both halves."""
    if "card_faces" in card:
        return "\n".join(f.get("oracle_text", "") for f in card["card_faces"])
    return card.get("oracle_text", "") or ""


def front(card):
    return card["card_faces"][0] if "card_faces" in card else card


def num(v):
    """Power/toughness may be '*', '1+*', None."""
    if v is None:
        return np.nan
    try:
        return float(v)
    except ValueError:
        m = re.match(r"(\d+)", str(v))
        return float(m.group(1)) if m else 0.0


def mana_symbols(cost):
    return re.findall(r"\{([^}]+)\}", cost or "")


def extract(card):
    f = {}
    t = faces(card).lower()
    tl = card.get("type_line", "").lower()
    fr = front(card)
    # Split/adventure/prepare cards carry a combined "{3}{W} // {G/W}" cost; the
    # front face is what you actually pay to put the permanent down.
    cost = (fr.get("mana_cost") if "card_faces" in card else card.get("mana_cost")) or \
           card.get("mana_cost") or ""
    cmc = card.get("cmc", 0.0)

    # --- cost / colour ---------------------------------------------------
    f["cmc"] = cmc
    syms = mana_symbols(cost)
    colored = [s for s in syms if re.search(r"[WUBRG]", s)]
    f["n_colored_pips"] = len(colored)
    f["n_colors"] = len(card.get("colors", fr.get("colors", [])) or [])
    f["is_gold"] = 1.0 if f["n_colors"] >= 2 else 0.0
    f["is_colorless"] = 1.0 if f["n_colors"] == 0 else 0.0
    f["is_hybrid"] = 1.0 if any("/" in s for s in colored) else 0.0
    f["colour_intensity"] = f["n_colored_pips"] / cmc if cmc else 0.0

    # --- rarity ----------------------------------------------------------
    for r in ("common", "uncommon", "rare", "mythic"):
        f[f"rarity_{r}"] = 1.0 if card.get("rarity") == r else 0.0
    f["rarity_ord"] = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}.get(card.get("rarity"), 0)

    # --- types -----------------------------------------------------------
    for ty in ("creature", "instant", "sorcery", "enchantment", "artifact",
               "land", "planeswalker", "aura", "equipment", "legendary", "battle"):
        f[f"type_{ty}"] = 1.0 if ty in tl else 0.0
    f["is_permanent"] = 1.0 if not (f["type_instant"] or f["type_sorcery"]) else 0.0
    f["is_dfc"] = 1.0 if "card_faces" in card else 0.0

    # --- body ------------------------------------------------------------
    p, tough = num(fr.get("power")), num(fr.get("toughness"))
    f["power"], f["toughness"] = p, tough
    f["pt_sum"] = (p + tough) if not (np.isnan(p) or np.isnan(tough)) else np.nan
    f["pt_per_mana"] = f["pt_sum"] / cmc if (cmc and not np.isnan(f["pt_sum"])) else np.nan
    f["power_per_mana"] = p / cmc if (cmc and not np.isnan(p)) else np.nan
    # a 2-mana 3/3 beats the curve; measures raw stat efficiency vs vanilla baseline
    f["stat_excess"] = (f["pt_sum"] - (2 * cmc + 1)) if not np.isnan(f["pt_sum"]) else np.nan

    # --- keywords --------------------------------------------------------
    kw = {k.lower() for k in card.get("keywords", [])}
    for k in EVERGREEN:
        f["kw_" + k.replace(" ", "_")] = 1.0 if (k in kw or k in t) else 0.0
    f["n_keywords"] = float(len(kw))
    f["evasive"] = 1.0 if (f["kw_flying"] or f["kw_menace"] or "can't be blocked" in t
                           or "unblockable" in t) else 0.0

    # --- effect classes (the meat) ---------------------------------------
    def has(*pats):
        return 1.0 if any(re.search(p, t) for p in pats) else 0.0

    # Removal is the single most predictive effect class in limited, so it gets
    # sub-classes rather than one flag. Patterns validated against WotC's own
    # published removal list (see validate_removal.py).
    CREA = r"(?:creature|planeswalker|permanent)"
    f["e_destroy_exile"]    = has(rf"\b(destroy|exile) (?:target|up to \w+ target|all|each|another target)[^.]{{0,70}}\b{CREA}")
    f["e_destroy_target"]   = has(rf"\b(destroy|exile) target[^.]{{0,70}}\b{CREA}")
    f["e_exile_target"]     = has(rf"\bexile target[^.]{{0,70}}\b{CREA}")
    f["e_uncond_removal"]   = has(r"\b(destroy|exile) target creature\b(?![^.]{0,50}\b(with|that|unless|if)\b)")
    f["e_cond_removal"]     = has(r"\b(destroy|exile) target[^.]{0,70}\b(with (mana value|power|toughness)|that's|unless|if)")
    # "deals N damage to any target" / "... to target <qualifiers> creature"
    f["e_damage_removal"]   = has(rf"deals? (?:\d+|x) damage to (?:any target|[^.]{{0,60}}\b{CREA})")
    f["e_debuff"]           = has(r"gets? -\d+/-\d+", r"base power and toughness 0/0",
                                  r"all creatures get -\d+/-\d+")
    f["e_neutralize"]       = has(r"loses all abilities", r"can't attack or block",
                                  r"doesn't untap", r"tap target creature", r"phases out")
    f["e_fight"]            = has(r"\bfights?\b", r"deals damage equal to its power to target")
    f["e_edict"]            = has(r"sacrifices? a creature")
    f["e_boardwipe"]        = has(r"destroy all", r"exile all",
                                  r"all creatures get -\d+/-\d+", r"each creature gets -\d+/-\d+")
    f["e_artifact_removal"] = has(r"(destroy|exile) target (artifact|enchantment)")
    f["e_counter_removal"]  = has(r"remove .{0,30}counters? from")
    f["e_hand_disruption"]  = has(r"reveals their hand", r"target (opponent|player) discards")
    f["e_bounce"]           = has(r"return target .{0,60}to (its owner's|their owner's|the owner's) hand",
                                  r"put target creature .{0,30}on top of (its owner's|their) library",
                                  r"on (their|its owner's|the owner's) choice of the top or bottom",
                                  r"puts? it on the (top|bottom) of (their|its owner's) library")
    f["e_counter"]          = has(r"counter target")
    f["e_tap_down"]         = f["e_neutralize"]
    # the headline flag: does this card answer an opposing threat at all?
    f["e_removal_any"]      = 1.0 if any(f[k] for k in
        ("e_destroy_exile", "e_damage_removal", "e_debuff", "e_neutralize",
         "e_fight", "e_edict", "e_boardwipe")) else 0.0
    f["e_removal_quality"]  = (2.0 * f["e_destroy_exile"] + 1.5 * f["e_damage_removal"]
                               + 1.0 * f["e_debuff"] + 1.0 * f["e_fight"]
                               + 0.5 * f["e_neutralize"] + 2.0 * f["e_boardwipe"])

    f["e_draw"]             = has(r"draw (a|\w+) cards?")
    f["e_draw_repeat"]      = has(r"whenever.{0,80}draw a card", r"at the beginning of.{0,60}draw")
    f["e_selection"]        = has(r"\bscry\b", r"\bsurveil\b", r"look at the top")
    f["e_tutor"]            = has(r"search your library")
    f["e_recursion"]        = has(r"return target .{0,40}from your graveyard",
                                  r"return .{0,30}card from your graveyard")
    f["e_token"]            = has(r"create .{0,50}token")
    f["e_token_multi"]      = has(r"create (two|three|four|five|x) ")
    f["e_counters"]         = has(r"\+1/\+1 counter")
    f["e_anthem"]           = has(r"creatures you control get \+")
    f["e_lifegain"]         = has(r"you gain \d+ life", r"gain \d+ life", r"lifelink")
    f["e_ramp"]             = has(r"add \{", r"search your library for a .{0,30}land",
                                  r"lands? .{0,20}onto the battlefield")
    f["e_discard"]          = has(r"discards? (a|\w+) cards?")
    f["e_mill"]             = has(r"mills? (a|\w+|\d+) cards?", r"put the top .{0,20}graveyard")
    f["e_sacrifice_outlet"] = has(r"sacrifice (a|another) creature:")
    f["e_reach_face"]       = has(r"deals? \d+ damage to (target player|each opponent|target opponent)")
    f["e_protection"]       = has(r"hexproof", r"indestructible", r"protection from",
                                  r"regenerate", r"\bward\b")
    f["e_gain_control"]     = has(r"gain control of")

    # --- triggers / timing ----------------------------------------------
    f["trig_etb"]      = has(r"when (this creature|this permanent|.{0,25}) enters")
    f["trig_death"]    = has(r"when .{0,30}dies")
    f["trig_attack"]   = has(r"whenever .{0,30}attacks")
    f["trig_upkeep"]   = has(r"at the beginning of (your|each) (upkeep|end step|precombat)")
    f["n_activated"]   = float(len(re.findall(r"[^\n]:\s", t)))
    f["has_activated"] = 1.0 if f["n_activated"] else 0.0
    f["is_instant_speed"] = 1.0 if (f["type_instant"] or f["kw_flash"]) else 0.0

    # --- text shape -------------------------------------------------------
    f["text_len"]   = float(len(t))
    f["n_lines"]    = float(t.count("\n") + 1) if t else 0.0
    f["is_vanilla"] = 1.0 if (f["type_creature"] and len(t) < 30 and not f["n_keywords"]) else 0.0

    # --- archetype breadth -------------------------------------------------
    # How many of the ten standard two-colour strategies does this card actively
    # serve? Written as generic rules patterns, so it works on any set. A card
    # that only ever does work in one archetype is a narrower pick than its raw
    # stats suggest; one that serves many is more reliably playable.
    from guards import ARCHETYPE_SIGNALS
    hits = [sum(1 for p_ in pats if re.search(p_, t)) for pats in ARCHETYPE_SIGNALS.values()]
    f["arch_breadth"] = float(sum(1 for h in hits if h))
    f["arch_depth"] = float(max(hits)) if hits else 0.0

    # --- efficiency interactions -----------------------------------------
    # cheap removal is the premium commodity in limited; encode it explicitly
    f["cheap_removal"]  = f["e_removal_any"] * max(0.0, 5.0 - cmc)
    f["cheap_interact"] = (f["e_removal_any"] or f["e_counter"] or f["e_bounce"]) * max(0.0, 5.0 - cmc)
    f["evasive_body"]   = f["evasive"] * (p if not np.isnan(p) else 0.0)
    return f


def feature_names(sample_card):
    return sorted(extract(sample_card))


def matrix(cards):
    rows = [extract(c) for c in cards]
    names = sorted(rows[0])
    X = np.array([[r.get(n, np.nan) for n in names] for r in rows], dtype=np.float64)
    return X, names
