"""BM25-подобный OCR-скор: OCR = запрос, карточки кандидатов = документы, IDF по пулу."""
import pickle
import numpy as np
import lab_eval as E

from lab_data import lab_cache

HERE = lab_cache()
FIELDS = {"n": 0, "p": 1, "g": 2}


def token_arrays(D, translit=True):
    toks = pickle.loads((HERE / "tokens.pkl").read_bytes())
    cand, field, s, df, npool, dlen, isnum = [], [], [], [], [], [], []
    for i, rec in enumerate(toks):
        cands = rec[translit]
        n = len(cands)
        counts = {}
        for ts in cands:
            for f, k, _ in ts:
                counts[(f, k)] = counts.get((f, k), 0) + 1
        for j, ts in enumerate(cands):
            L = sum(1 for f, _, _ in ts if f == "n")
            for f, k, sc in ts:
                cand.append(D.starts[i] + j); field.append(FIELDS[f]); s.append(sc)
                df.append(counts[(f, k)]); npool.append(n); dlen.append(L)
                isnum.append(f != "g" and k.isdigit())
    A = lambda x, t=np.float64: np.array(x, dtype=t)
    return dict(cand=A(cand, np.int64), field=A(field, np.int64), s=A(s), df=A(df), n=A(npool),
                dlen=A(dlen), isnum=A(isnum, bool))


BM_SPACE = {
    **E.PARAM_SPACE,
    "tau": (0.6, 0.95),
    "alpha": (0.0, 2.0),      # степень IDF (0 = без IDF)
    "b": (0.0, 1.0),          # нормировка на длину названия
    "num_w": (0.0, 1.0),      # вес числовых токенов (год и т.п.)
    "w_bm": (0.0, 0.6),
}


def make_bm25_scorer(D, T, use_old=False, matcher="win"):
    base = E.make_scorer(D, matcher, "", use_extra=True)
    ncand = D.X.shape[0]
    idf = np.log((T["n"] - T["df"] + 0.5) / (T["df"] + 0.5) + 1.0)
    avg = np.maximum(T["dlen"].mean(), 1.0)

    def score(p):
        soft = np.clip((T["s"] - p["tau"]) / (1 - p["tau"]), 0, 1)
        w = idf ** p["alpha"] * soft
        fw = np.array([p["w_name"], p["w_prod"], p["w_grape"]])[T["field"]]
        norm_len = np.where(T["field"] == 0, 1 - p["b"] + p["b"] * T["dlen"] / avg, 1.0)
        w = w * fw / norm_len * np.where(T["isnum"], p["num_w"], 1.0)
        text = np.bincount(T["cand"], weights=w, minlength=ncand)
        q = dict(p)
        if not use_old:
            q.update(lam=0.0)  # старые entity-признаки выключены, остаются противоречия
        return base(q) + p["w_bm"] * text

    return score
