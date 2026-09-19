"""Parse WotC's prerelease guide into structured set knowledge.

The guide gives us three things no statistical model could infer from card text
alone: the official removal list, the ten draft archetypes, and the key cards
WotC flags as archetype payoffs/enablers. Card names come from image alt text
(e.g. "0015_MTGFRA_MainUP: Memory Trap"), which is stable and unambiguous.
"""
import re, html, json, sys, os

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "setinfo")

COLOR_LETTER = {"White": "W", "Blue": "U", "Black": "B", "Red": "R", "Green": "G"}
WUBRG = "WUBRG"


def clean_name(raw):
    """Strip the colour wording each layout leaves in the heading text."""
    C = r"(?:White|Blue|Black|Red|Green)"
    n = raw.replace("&nbsp;", " ")
    n = re.sub(rf"^\s*{C}-{C}\s+", "", n)          # "White-Blue Second Spell"
    n = re.sub(rf"\s*\(\s*{C}-{C}\s*\)\s*$", "", n)   # "Eerie Tempo (White-Blue)"
    return re.sub(r"\s+", " ", n).strip().strip("—-–").strip()


def detect_archetypes(dec):
    """Read the ten archetypes off the guide's own headings.

    Every modern prerelease guide names them the same way -- "<Colour>-<Colour>
    <Archetype Name>" as a section heading -- so this works on any set's guide
    rather than needing the archetypes typed in per set.
    """
    seen, out = set(), []
    C = r"(White|Blue|Black|Red|Green)"
    patterns = [
        # "White-Blue Fatehold Surveil Aggro"   (FRA, EOE, TLA, ECL ...)
        (rf"<h[1-6][^>]*>\s*{C}-{C}\s+([^<]{{2,60}})", "prefix"),
        # "Birdfolk (White-Blue)"               (BLB and friends)
        (rf"<h[1-6][^>]*>\s*([^<(]{{2,60}}?)\s*\(\s*{C}-{C}\s*\)", "suffix"),
    ]
    # Third layout: the heading carries the name, and the colours arrive as mana
    # symbol images right after it -- "<h3>Silverquill Repartee Aggro <img
    # alt='White mana symbol'><img alt='Black mana symbol'></h3>" (SOS, FRA).
    sym = r"<img[^>]*alt=\"(White|Blue|Black|Red|Green) mana symbol\"[^>]*>"
    for m in re.finditer(rf"<h[1-6][^>]*>\s*([^<]{{2,60}}?)\s*({sym}\s*){{2}}", dec):
        head = dec[m.start():m.end()]
        cols = re.findall(sym, head)
        if len(cols) < 2:
            continue
        a, b = COLOR_LETTER[cols[0]], COLOR_LETTER[cols[1]]
        pair = "".join(sorted({a, b}, key=WUBRG.index))
        name = clean_name(m.group(1))
        if pair in seen or not name:
            continue
        seen.add(pair)
        out.append((pair, name, ""))

    for pat, kind in patterns:
        for m in re.finditer(pat, dec):
            if kind == "prefix":
                a, b, name = COLOR_LETTER[m.group(1)], COLOR_LETTER[m.group(2)], m.group(3)
            else:
                name, a, b = m.group(1), COLOR_LETTER[m.group(2)], COLOR_LETTER[m.group(3)]
            pair = "".join(sorted({a, b}, key=WUBRG.index))
            name = clean_name(name)
            if pair in seen or not name:
                continue
            seen.add(pair)
            out.append((pair, name, ""))
    return out


def text_lines(raw):
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    t = html.unescape(re.sub(r"(?is)<[^>]+>", "\n", t))
    return [l.strip() for l in t.split("\n") if l.strip() and len(l.strip()) > 2]


def card_refs(raw):
    """Every card the guide names, in document order.

    The page embeds part of its markup as escaped JSON (\\u003C for '<'), so we
    decode first. Cards appear both as <cig-card> image tiles and as <auto-card>
    inline mentions in prose -- the prose ones matter, since that is where WotC
    explains what each archetype actually wants.
    """
    dec = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), raw)
    out = []
    for m in re.finditer(r"<(?:auto|cig)-card[^>]*>([^<]{2,70})</(?:auto|cig)-card>", dec):
        n = re.sub(r"^\d{3,4}_MTG[A-Z]{3}_[A-Za-z]+:\s*", "", m.group(1)).strip()
        n = re.sub(r"'s$", "", n).strip()
        if n and not re.search(r"\btokens?\b", n, re.I):
            out.append((m.start(), n))
    for m in re.finditer(r'auto-card name="([^"]{2,70})"', dec):
        n = re.sub(r"'s$", "", m.group(1)).strip()
        if n and not re.search(r"\btokens?\b", n, re.I):
            out.append((m.start(), n))
    return sorted(set(out)), dec


def parse(path, setcode):
    raw = open(path, encoding="utf8", errors="replace").read()
    refs, dec = card_refs(raw)
    archs = detect_archetypes(dec)
    if len(archs) < 5:
        raise SystemExit(f"only found {len(archs)} archetype headings in this guide; "
                         "its layout differs -- inspect before trusting the parse")

    # Removal: everything named between the removal prompt and the curve section.
    # Not every guide carries this block, so it degrades to an empty list.
    seen, removal = set(), []
    rs = next((dec.find(t) for t in ("key removal spells", "key removal")
               if dec.find(t) >= 0), -1)
    re_ = next((dec.find(t) for t in ("Deck Building with the Mana Curve", "Mana Curve")
                if dec.find(t) > rs >= 0), -1)
    if rs >= 0 and re_ > rs:
        for pos, n in refs:
            if rs < pos < re_ and n not in seen:
                seen.add(n); removal.append(n)

    # Archetypes: each block runs from its own heading to the next one.
    bounds = []
    for colors, name, hook in archs:
        i = dec.find(name)
        bounds.append((i if i >= 0 else 10 ** 9, colors, name, hook))
    bounds.sort()
    archetypes = {}
    for k, (start, colors, name, hook) in enumerate(bounds):
        if k + 1 < len(bounds):
            end = bounds[k + 1][0]
        else:
            # last archetype: stop at the wrap-up prose, or the whole page tail
            # (footer, related articles, embedded JSON) lands in this block
            # The final archetype runs to the end of the document unless bounded,
            # which drags the footer and related-article cards into it. The next
            # <h2> is the reliable boundary; prose markers are a fallback.
            nxt = re.search(r"<h2[\s>]", dec[start + 1:])
            tail = [start + 1 + nxt.start()] if nxt else []
            tail += [dec.find(t, start) for t in
                     ("Card Image Gallery", "Prerelease events begin on",
                      "brings the greatest pieces")]
            tail = [t for t in tail if t > start]
            end = min(tail) if tail else len(dec)
        keys = sorted({n for pos, n in refs if start <= pos < end}) if start < 10 ** 9 else []
        archetypes[colors] = {"name": name, "hook": hook, "key_cards": keys}

    return {"set": setcode, "removal": removal,
            "archetypes": {c: archetypes[c] for c, _, _ in archs}}


if __name__ == "__main__":
    res = parse(sys.argv[1], sys.argv[2])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{sys.argv[2]}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"removal spells: {len(res['removal'])}")
    for c, a in res["archetypes"].items():
        print(f"  {c:2s} {a['name']:28s} {len(a['key_cards'])} key cards")
