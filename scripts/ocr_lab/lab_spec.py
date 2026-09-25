"""Признак "уникальные доказательства": sum IDF_pool найденных в OCR слов названия+производителя."""
import pickle
import numpy as np
import lab_eval as E

from lab_data import lab_cache

HERE = lab_cache()


def spec_arrays(D, translit=True):
    toks = pickle.loads((HERE / "tokens.pkl").read_bytes())
    cand, s, idf, isnum = [], [], [], []
    for i, rec in enumerate(toks):
        cands = rec[translit]
        n = len(cands)
        docs = []
        for ts in cands:
            d = {}
            for f, k, x in ts:
                if f in ("n", "p"):
                    d[k] = max(d.get(k, 0.0), x)
            docs.append(d)
        df = {}
        for d in docs:
            for k in d:
                df[k] = df.get(k, 0) + 1
        for j, d in enumerate(docs):
            for k, x in d.items():
                cand.append(D.starts[i] + j); s.append(x)
                idf.append(np.log((n - df[k] + 0.5) / (df[k] + 0.5) + 1.0)); isnum.append(k.isdigit())
    return dict(cand=np.array(cand), s=np.array(s), idf=np.array(idf), isnum=np.array(isnum))


def make_scorer(D, S, matcher="win", suffix=""):
    base = E.make_scorer(D, matcher, suffix)
    ncand = D.X.shape[0]

    def score(p):
        ok = (S["s"] >= p["spec_tau"]) & ~S["isnum"]
        spec = np.bincount(S["cand"], weights=S["idf"] * ok, minlength=ncand)
        return base(p) + p["w_spec"] * spec

    return score


SPACE = {**E.PARAM_SPACE, "w_spec": (0.0, 0.03), "spec_tau": (0.75, 0.95)}
