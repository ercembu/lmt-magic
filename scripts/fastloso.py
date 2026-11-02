"""Leave-one-set-out, fast enough to iterate with.

An experiment used to take ~30 minutes, and almost none of that was the model:

  * the text vectoriser was refit for every fold of every sweep, although it
    depends only on which set is held out -- so it is fitted once per fold here
    and cached on disk;
  * the 32 folds ran one after another with LightGBM on 8 threads, and a fit on
    ~9k cards does not scale across threads -- so folds run in parallel, each on
    a couple of threads;
  * every experiment recomputed the same baseline before its variant -- so
    baselines are cached on disk, keyed on everything that determines them.

Same model, same folds, same numbers as tune.run (up to LightGBM's thread-count
rounding). A variant is a function of the held-out set that returns extra dense
columns, so fold-aware features (cross-fitted scores) work too.
"""
import hashlib, os, pickle, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
import dataset, textfeat, train

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "cache")
N_JOBS = int(os.environ.get("LOSO_JOBS", 8))
THREADS = max(1, (os.cpu_count() or 16) // N_JOBS)


def _key(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p.tobytes() if isinstance(p, np.ndarray) else repr(p).encode())
    return h.hexdigest()[:16]


def _cached(name, key, make):
    # Pickle is safe here: these files are written by this script into the local
    # data/cache/ and never come from anywhere else. Delete the folder to reset.
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}-{key}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    out = make()
    pickle.dump(out, open(path, "wb"))
    return out


class Folds:
    """Everything about a LOSO split that does not depend on the variant."""

    def __init__(self, min_cards=30):
        self.sets = [s for s in train.TRAIN_SETS
                     if os.path.exists(f"{ROOT}/data/labels/{s}.json")]
        self.X, self.y, self.texts, self.w, self.rows = dataset.build(self.sets, verbose=False)
        self.arr = np.array([r["set"] for r in self.rows])
        self.held = [s for s in self.sets if (self.arr == s).sum() >= min_cards]
        tkey = _key(np.array(self.texts), self.arr)
        t0 = time.time()
        self.text = _cached("foldtext", tkey, lambda: dict(Parallel(n_jobs=N_JOBS)(
            delayed(_fold_text)(self.texts, self.arr, s) for s in self.held)))
        print(f"{len(self.held)} folds, {len(self.y)} cards "
              f"(text blocks ready in {time.time() - t0:.0f}s)", flush=True)

    def dense(self, blind=False):
        X = self.X.copy()
        if blind:
            X[:, dataset.FEATURE_KEYS.index("expert_rating")] = np.nan
        return X

    def sweep(self, X, extra=None, extra_names=(), w=None, params=None, label=None):
        """Per-set spearman for one configuration. `extra(s)` -> (n, k) dense columns."""
        w = self.w if w is None else w
        p = dict(train.PARAMS, num_threads=THREADS, **(params or {}))
        cfg = (p, train.ROUNDS, train.ENSEMBLE_GBM_WEIGHT, train.RIDGE_ALPHA)
        run = lambda: dict(Parallel(n_jobs=N_JOBS)(
            delayed(_fit_fold)(s, X, extra(s) if extra else None, list(extra_names),
                               self.y, w, self.arr, self.text[s], cfg)
            for s in self.held))
        t0 = time.time()
        if extra is None:          # only plain configurations are safe to cache by value
            out = _cached("sweep", _key(X, self.y, w, cfg), run)
        else:
            out = run()
        if label:
            print(f"  {label:38s} {np.mean(list(out.values())):.4f}   "
                  f"({time.time() - t0:.0f}s)", flush=True)
        return out


def _fold_text(texts, arr, s):
    te = arr == s
    vec = textfeat.vectorizer()
    ttr = [t for t, m in zip(texts, ~te) if m]
    tte = [t for t, m in zip(texts, te) if m]
    Ttr, Tte = vec.fit_transform(ttr), vec.transform(tte)
    names = [f"tf_{g}" for g in vec.get_feature_names_out()]
    return s, (Ttr.tocsr(), Tte.tocsr(), train.novelty(ttr, vec), train.novelty(tte, vec), names)


def _fit_fold(s, X, E, enames, y, w, arr, text, cfg):
    params, rounds, gw, alpha = cfg
    Ttr, Tte, Ntr, Nte, tnames = text
    te = arr == s
    tr = ~te
    D = X if E is None else np.hstack([X, E])
    def block(m, N, T):
        return sp.hstack([sp.csr_matrix(np.nan_to_num(D[m], nan=-999)),
                          sp.csr_matrix(N), T]).tocsr()
    names = [n.replace(" ", "_") for n in
             dataset.FEATURE_KEYS + enames + ["oov_rate", "mean_idf"] + tnames]
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    from scipy.stats import rankdata
    Xtr, Xte = block(tr, Ntr, Ttr), block(te, Nte, Tte)
    gbm = lgb.train(params, lgb.Dataset(Xtr, label=y[tr], weight=w[tr], feature_name=names),
                    num_boost_round=rounds)
    ridge = Ridge(alpha=alpha).fit(Xtr, y[tr], sample_weight=w[tr])
    g, r = gbm.predict(Xte), ridge.predict(Xte)
    n = len(g)
    pred = gw * rankdata(g) / n + (1 - gw) * rankdata(r) / n
    return s, float(spearmanr(pred, y[te]).statistic)


if __name__ == "__main__":
    f = Folds()
    for mode in ("blind", "sighted"):
        f.sweep(f.dense(blind=mode == "blind"), label=f"{mode} baseline")
