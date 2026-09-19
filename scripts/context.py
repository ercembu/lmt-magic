"""Set-context features: how a card compares to the set it lives in.

The model's weakest point is commons (spearman 0.443, against 0.608 on mythics),
and commons are exactly what decides whether a colour is deep. The reason is that
a common is only good or bad *relative to its format*: a 3/3 for three is fine in
a slow set and unplayable in a fast one, and nothing in the per-card features says
which kind of set this is.

Everything here is computed from card data only -- no win rates -- so it is
available for an unreleased set exactly as it is for a released one. The two
strongest ideas:

  * body vs the curve *in this set*: how a creature's stats compare to other
    creatures of the same mana value in the same set.
  * survivability: whether a creature's toughness beats the damage the set's own
    common removal actually deals.
"""
import re
import numpy as np


def _dmg_numbers(text):
    return [int(x) for x in re.findall(r"deals? (\d+) damage", text or "")]


def set_profile(cards, feats):
    """Format-level summary of a set, from its cards alone."""
    cre = [f for f in feats if f.get("type_creature")]
    pt = [f["pt_per_mana"] for f in cre
          if f.get("pt_per_mana") is not None and not np.isnan(f["pt_per_mana"])]
    cmcs = [f["cmc"] for f in feats if f["cmc"] > 0]

    # what the set's cheap removal actually kills
    dmg = []
    for c, f in zip(cards, feats):
        if not f.get("e_damage_removal") or c.get("rarity") not in ("common", "uncommon"):
            continue
        txt = c.get("oracle_text") or ""
        if "card_faces" in c:
            txt = "\n".join(x.get("oracle_text", "") or "" for x in c["card_faces"])
        dmg += _dmg_numbers(txt.lower())

    # average body at each mana value -- the set's own creature curve
    curve = {}
    for f in cre:
        mv = int(min(f["cmc"], 8))
        if f.get("pt_sum") is not None and not np.isnan(f["pt_sum"]):
            curve.setdefault(mv, []).append(f["pt_sum"])
    curve = {k: float(np.mean(v)) for k, v in curve.items() if len(v) >= 3}

    return {
        "mean_cmc": float(np.mean(cmcs)) if cmcs else 3.0,
        "mean_pt_per_mana": float(np.mean(pt)) if pt else 1.0,
        "removal_density": float(np.mean([f["e_removal_any"] for f in feats])),
        "cheap_removal_density": float(np.mean(
            [f["e_removal_any"] for c, f in zip(cards, feats)
             if c.get("rarity") in ("common", "uncommon")] or [0.0])),
        "evasion_density": float(np.mean([f["evasive"] for f in feats])),
        "creature_density": float(np.mean([f["type_creature"] for f in feats])),
        "mean_text_len": float(np.mean([f["text_len"] for f in feats])),
        "median_burn": float(np.median(dmg)) if dmg else 3.0,
        "curve": curve,
    }


COLOR_METRICS = ["removal", "pt_per_mana", "stat_excess", "evasion",
                 "cheap_interact", "body_vs_curve", "text_len", "keywords"]

# Each colour metric appears twice: the colour's own level, and how that level
# compares to the rest of the set. The comparison is the point -- it is the only
# thing in the feature set that says "your colour is the weak one here".
COLOR_KEYS = ([f"col_{m}" for m in COLOR_METRICS]
              + [f"col_{m}_vs_set" for m in COLOR_METRICS])


def color_profiles(cards, feats, prof):
    """Per-colour summaries of a set, from card data only.

    The model's errors cluster by colour: in a given set it tends to misjudge a
    whole colour in one direction, because nothing tells it how this set's
    colours were balanced against each other. These features do.
    """
    def metrics(sel):
        if not sel:
            return {m: np.nan for m in COLOR_METRICS}
        def mean(key, default=np.nan):
            v = [f[key] for f in sel
                 if f.get(key) is not None and not np.isnan(f.get(key, np.nan))]
            return float(np.mean(v)) if v else default
        cre = [f for f in sel if f.get("type_creature")]
        return {
            "removal": float(np.mean([f["e_removal_any"] for f in sel])),
            "pt_per_mana": mean("pt_per_mana"),
            "stat_excess": mean("stat_excess"),
            "evasion": float(np.mean([f["evasive"] for f in sel])),
            "cheap_interact": float(np.mean([f["cheap_interact"] for f in sel])),
            "body_vs_curve": float(np.mean(
                [f["pt_sum"] - prof["curve"].get(int(min(f["cmc"], 8)), f["pt_sum"])
                 for f in cre if f.get("pt_sum") is not None
                 and not np.isnan(f["pt_sum"])] or [np.nan])),
            "text_len": mean("text_len"),
            "keywords": mean("n_keywords"),
        }

    overall = metrics(feats)
    out = {}
    for col in "WUBRG":
        sel = [f for c, f in zip(cards, feats) if col in (c.get("colors") or [])]
        out[col] = metrics(sel)
    out["_set"] = overall
    return out


def color_features(card, profiles):
    """Attach the card's own colours' profile, relative to the set."""
    cols = card.get("colors") or []
    st = profiles["_set"]
    if not cols:                       # colourless: the set average is its context
        mine = st
    else:
        mine = {m: float(np.nanmean([profiles[c][m] for c in cols if c in profiles]))
                for m in COLOR_METRICS}
    o = {}
    for m in COLOR_METRICS:
        o[f"col_{m}"] = mine[m]
        o[f"col_{m}_vs_set"] = (mine[m] - st[m]
                                if not (np.isnan(mine[m]) or np.isnan(st[m])) else np.nan)
    return o


KEYS = ["ctx_cmc_vs_set", "ctx_pt_vs_set", "ctx_body_vs_curve", "ctx_survives_burn",
        "ctx_text_vs_set", "ctx_removal_density", "ctx_cheap_removal_density",
        "ctx_evasion_density", "ctx_creature_density", "ctx_mean_cmc",
        "ctx_median_burn", "ctx_outguns_curve"] + COLOR_KEYS


def contextual(card, f, prof):
    """Per-card features that only mean anything relative to the set."""
    o = {}
    o["ctx_cmc_vs_set"] = f["cmc"] - prof["mean_cmc"]
    ppm = f.get("pt_per_mana")
    o["ctx_pt_vs_set"] = ((ppm - prof["mean_pt_per_mana"])
                          if ppm is not None and not np.isnan(ppm) else np.nan)
    o["ctx_text_vs_set"] = f["text_len"] - prof["mean_text_len"]

    mv = int(min(f["cmc"], 8))
    base = prof["curve"].get(mv)
    pt_sum = f.get("pt_sum")
    has_body = pt_sum is not None and not np.isnan(pt_sum)
    # is this a bigger body than the set's other creatures at the same cost?
    o["ctx_body_vs_curve"] = (pt_sum - base) if (base and has_body) else np.nan
    o["ctx_outguns_curve"] = float(pt_sum > base) if (base and has_body) else np.nan

    tough = f.get("toughness")
    o["ctx_survives_burn"] = (float(tough > prof["median_burn"])
                              if tough is not None and not np.isnan(tough) else np.nan)

    for k in ("removal_density", "cheap_removal_density", "evasion_density",
              "creature_density", "mean_cmc", "median_burn"):
        o["ctx_" + k] = prof[k]
    return o
