"""Find the model's blind spots by name, using the reviewers' own words.

We have ~7,000 written card evaluations and, for the same cards, how wrong the
model was. Regressing the model's error on the prose vocabulary names the
concepts it systematically mishandles -- instead of guessing at features, ask
what reviewers say about the cards we get wrong.

A term with a positive weight marks cards the model OVER-rates; negative marks
ones it UNDER-rates.
"""
import json, os, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(__file__))
import predict

ROOT = os.path.join(os.path.dirname(__file__), "..")
STOP = ["the","a","an","and","or","of","to","is","it","this","that","in","for","you","your",
        "with","on","as","be","are","if","but","not","have","has","will","can","its","at",
        "very","just","really","quite","also","get","gets","there","they","them","one","two"]


def load():
    prose = json.load(open(f"{ROOT}/data/prose.json"))
    oof = json.load(open(f"{ROOT}/data/oof.json"))
    by = {(r["set"], r["name"]): r for r in oof}
    # reviewer names are the printed name; match loosely like elsewhere
    idx = {}
    for (s, n), r in by.items():
        idx[(s, n)] = r
        idx[(s, n.split(" // ")[0])] = r
        idx[(s, n.split(",")[0])] = r
    rows = []
    for p in prose:
        r = idx.get((p["set"], p["name"]))
        if r and r.get("gih_wr"):
            rows.append((p, r))
    return rows


if __name__ == "__main__":
    rows = load()
    print(f"matched {len(rows)} write-ups to out-of-fold predictions")
    if len(rows) < 500:
        raise SystemExit("not enough overlap")

    # residual per set, on a common scale: positive = model rated it too highly
    sets = {p["set"] for p, _ in rows}
    resid = np.zeros(len(rows))
    for s in sets:
        ix = [i for i, (p, _) in enumerate(rows) if p["set"] == s]
        if len(ix) < 30:
            continue
        pz = predict.rank_to_z([rows[i][1]["pred"] for i in ix])
        az = predict.rank_to_z([rows[i][1]["gih_wr"] for i in ix])
        for j, i in enumerate(ix):
            resid[i] = pz[j] - az[j]

    texts = [p["prose"].lower() for p, _ in rows]
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=25, max_features=4000,
                          sublinear_tf=True, stop_words=STOP)
    T = vec.fit_transform(texts)
    m = Ridge(alpha=1.0).fit(T, resid)
    names = vec.get_feature_names_out()
    order = np.argsort(m.coef_)

    # NOTE: this is IN-SAMPLE and badly inflated -- 4,000 features on ~5,900 rows.
    # Leave-one-set-out puts the real figure at 0.6%, i.e. the prose does not
    # generalise across sets: the strongest terms are set-specific nouns
    # (treasure, ring, oil) rather than transferable ideas. The terms below are
    # worth reading as a diagnostic, not trusted as a model.
    print(f"\nprose explains {m.score(T, resid):.1%} of the variance in our error "
          f"(IN-SAMPLE; out-of-sample is 0.6% -- see README)\n")
    print("PHRASES ON CARDS WE RATE TOO HIGH  (reviewer sees a problem we miss)")
    for i in order[::-1][:18]:
        print(f"   {m.coef_[i]:+.3f}  {names[i]}")
    print("\nPHRASES ON CARDS WE RATE TOO LOW  (reviewer sees value we miss)")
    for i in order[:18]:
        print(f"   {m.coef_[i]:+.3f}  {names[i]}")
