"""Oracle-text vectorisation.

Reminder text (the italics in parentheses) is stripped: it restates rules the
model already sees elsewhere, inflates length, and names set-specific mechanics
that will never appear in a set the model is asked to grade cold.

Card names are replaced with '~' and numbers with 'N' so the vectoriser learns
effect vocabulary ("destroy target creature") rather than proper nouns, which
never transfer across sets.
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize(card):
    if "card_faces" in card:
        txt = "\n".join(f.get("oracle_text", "") or "" for f in card["card_faces"])
        names = [f["name"] for f in card["card_faces"]] + [card["name"]]
    else:
        txt = card.get("oracle_text", "") or ""
        names = [card["name"]]
    txt = re.sub(r"\([^)]*\)", " ", txt)              # reminder text
    for n in sorted(names, key=len, reverse=True):
        txt = txt.replace(n, "~")
        if "," in n:
            txt = txt.replace(n.split(",")[0], "~")
    txt = txt.lower()
    # lowercase placeholders: the final strip below removes anything else
    txt = re.sub(r"\{[^}]*\}", " manasym ", txt)
    txt = re.sub(r"[+-]?\d+/[+-]?\d+", " ptval ", txt)
    txt = re.sub(r"\b\d+\b", " numval ", txt)
    txt = re.sub(r"[^a-z~ ]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


# Grammatical filler that appears on nearly every card. Left in, it dominates the
# vectoriser and just re-encodes text length, which the dense features already have.
STOP = ["a", "an", "the", "of", "to", "and", "or", "it", "its", "is", "be", "as",
        "at", "in", "on", "for", "from", "that", "this", "these", "those", "with",
        "you", "your", "their", "them", "they", "he", "she", "up", "may", "if",
        "then", "than", "when", "whenever", "until", "end", "turn", "other",
        "another", "one", "each", "any", "all", "have", "has", "had", "do", "does"]


def vectorizer():
    """Effect vocabulary only -- filler words are stopped out."""
    return TfidfVectorizer(ngram_range=(1, 3), min_df=10, max_features=2500,
                           sublinear_tf=True, strip_accents="unicode",
                           stop_words=STOP)
