"""Sentence-embedding view of rules text.

TF-IDF sees "destroy target creature" and "destroy target creature with mana
value 3 or less" as near-identical (shared n-grams) when they are very different
cards. A sentence encoder separates them: cosine 0.70 rather than ~1.0.

The encoder is pretrained and frozen, so embedding a card uses no win-rate data
and leaks nothing across folds.
"""
import json, os, re, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = os.path.join(ROOT, "data", "embeddings.npz")


def readable(card):
    """Light normalisation: keep real English, drop only reminder text and names.

    The TF-IDF pipeline replaces numbers and mana symbols with tokens, which
    helps a bag-of-words model and hurts a language model.
    """
    if "card_faces" in card:
        txt = "\n".join(f.get("oracle_text", "") or "" for f in card["card_faces"])
        names = [f["name"] for f in card["card_faces"]] + [card["name"]]
    else:
        txt = card.get("oracle_text", "") or ""
        names = [card["name"]]
    txt = re.sub(r"\([^)]*\)", " ", txt)
    for n in sorted(names, key=len, reverse=True):
        txt = txt.replace(n, "this card")
        if "," in n:
            txt = txt.replace(n.split(",")[0], "this card")
    head = f"{card.get('type_line','')}. {card.get('mana_cost','')}."
    pt = card.get("power")
    if pt is not None:
        head += f" {card.get('power')}/{card.get('toughness')}."
    return re.sub(r"\s+", " ", f"{head} {txt}").strip()


def build(setcodes):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL)
    keys, texts = [], []
    for s in setcodes:
        p = f"{ROOT}/data/cards/{s}.json"
        if not os.path.exists(p):
            continue
        for c in json.load(open(p)):
            keys.append(f"{s}||{c['name']}")
            texts.append(readable(c))
    vecs = m.encode(texts, batch_size=128, show_progress_bar=False,
                    normalize_embeddings=True)
    np.savez_compressed(CACHE, keys=np.array(keys), vecs=vecs.astype(np.float32))
    return keys, vecs


def load():
    d = np.load(CACHE, allow_pickle=False)
    return {k: i for i, k in enumerate(d["keys"])}, d["vecs"]


if __name__ == "__main__":
    import train
    sets = [s for s in train.TRAIN_SETS if os.path.exists(f"{ROOT}/data/cards/{s}.json")]
    sets = sets + ["FRA"]
    k, v = build(sets)
    print(f"embedded {len(k)} cards from {len(sets)} sets -> {v.shape}")
