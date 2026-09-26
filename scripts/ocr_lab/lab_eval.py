"""CV-оценка формул OCR-реранка (этап 1, без CatBoost)."""

from __future__ import annotations

import pickle
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np

from lab_data import lab_cache, load_catalog, load_records



HERE = lab_cache()
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ данные


def _prod_scores(rec: dict) -> list[float]:
    """OCR-бонус продовой формулы (src/ml/ocr_matching.py): final = visual + bonus."""
    from eval_formula import bonuses

    return bonuses(rec)


_CAT = load_catalog()


def load():
    records = load_records()
    feats = pickle.loads((HERE / "features.pkl").read_bytes())
    prod_path = HERE / "prod.pkl"
    if prod_path.exists():
        prod = pickle.loads(prod_path.read_bytes())
    else:
        with Pool(8) as pool:
            prod = pool.map(_prod_scores, records, chunksize=4)
        prod_path.write_bytes(pickle.dumps(prod))
    names = list(feats[0][0].keys())
    X = np.array([[f[n] for n in names] for fs in feats for f in fs], dtype=np.float64)
    col = {n: i for i, n in enumerate(names)}
    rec_idx = np.concatenate([np.full(len(fs), i) for i, fs in enumerate(feats)])
    starts = np.concatenate([[0], np.cumsum([len(fs) for fs in feats])[:-1]])
    label = np.array(
        [c["wine_id"] == r["expected"] for r in records for c in r["cands"]], dtype=bool
    )
    prod_arr = np.array([s for ps in prod for s in ps])
    groups = np.array([r["expected"] for r in records])
    return SimpleNamespace(X=X, col=col, rec_idx=rec_idx, starts=starts, label=label,
                           prod=prod_arr, groups=groups, n=len(records), records=records)


# ------------------------------------------------------------------ метрики


def metrics(D, score: np.ndarray, rec_mask: np.ndarray | None = None) -> tuple[float, float]:
    """Acc@1 и MRR; ничьи считаются пессимистично (правильный — ниже равных)."""
    exp_score = np.full(D.n, -np.inf)
    exp_score[D.rec_idx[D.label]] = score[D.label]
    better = np.bincount(D.rec_idx, weights=(score >= exp_score[D.rec_idx]) & ~D.label, minlength=D.n)
    rank = better + 1
    if rec_mask is not None:
        rank = rank[rec_mask]
    return float(np.mean(rank == 1)), float(np.mean(1.0 / rank))


# ------------------------------------------------------------------ формула этапа 1


def bonus(s: np.ndarray, tau: float) -> np.ndarray:
    """Только бонус за найденное: ниже tau — 0 (ничего не штрафуем за отсутствие)."""
    return np.clip((s - tau) / (1.0 - tau), 0.0, 1.0) * (s >= 0)


PARAM_SPACE = {
    "lam": (0.0, 0.6),
    "w_name": (0.0, 1.0), "w_prod": (0.0, 1.0), "w_grape": (0.0, 1.0),
    "tau": (0.3, 0.85),
    "grape_mix": (0.0, 1.0),       # mean vs max по сортам
    "c_bonus": (0.0, 0.1), "c_pen": (0.0, 0.15),
    "s_bonus": (0.0, 0.1), "s_pen": (0.0, 0.15),
    "g_pen": (0.0, 0.15), "p_pen": (0.0, 0.15),
    "alc": (0.0, 0.05),
}


def make_scorer(D, matcher: str, suffix: str = "", use_extra: bool = True):
    X, c = D.X, D.col
    name = X[:, c[f"name_{matcher}{suffix}"]]
    prod = X[:, c[f"producer_{matcher}{suffix}"]]
    g_mean = X[:, c[f"grape_{matcher}_mean{suffix}"]]
    g_max = X[:, c[f"grape_{matcher}_max{suffix}"]]
    vis = X[:, c["visual"]]
    c_own, c_oth = X[:, c["c_own"]], X[:, c["c_other"]]
    s_own, s_oth = X[:, c["s_own"]], X[:, c["s_other"]]
    g_extra, g_hit = X[:, c["g_extra"]], X[:, c["g_hit"]]
    p_other = X[:, c["p_other"]]
    alc_m, alc_x = X[:, c["alc_match"]], X[:, c["alc_miss"]]
    T = 0.8
    c_match, c_contra = (c_own >= T), (c_own < T) & (c_oth >= T) & (c_own >= 0)
    s_match, s_contra = (s_own >= T), (s_own < T) & (s_oth >= T) & (s_own >= 0)
    g_contra = (g_extra > 0) & (g_hit == 0) & (g_mean >= 0)

    def score(p: dict) -> np.ndarray:
        grape = p["grape_mix"] * g_mean + (1 - p["grape_mix"]) * g_max
        grape = np.where(g_mean < 0, -1.0, grape)
        text = (p["w_name"] * bonus(name, p["tau"]) + p["w_prod"] * bonus(prod, p["tau"])
                + p["w_grape"] * bonus(grape, p["tau"]))
        s = vis + p["lam"] * text
        if use_extra:
            s = (s + p["c_bonus"] * c_match - p["c_pen"] * c_contra
                 + p["s_bonus"] * s_match - p["s_pen"] * s_contra
                 - p["g_pen"] * g_contra - p["p_pen"] * p_other
                 + p["alc"] * (alc_m - alc_x))
        return s

    return score


def sample(n: int, space=PARAM_SPACE) -> list[dict]:
    return [{k: RNG.uniform(*v) for k, v in space.items()} for _ in range(n)]


def objective(D, score, mask):
    a, m = metrics(D, score, mask)
    return a + 0.5 * m


def fit(D, scorer, train_mask, n_random=1500, rounds=2, space=PARAM_SPACE):
    best_p, best_v = None, -1
    for p in sample(n_random, space):
        v = objective(D, scorer(p), train_mask)
        if v > best_v:
            best_p, best_v = p, v
    for _ in range(rounds):  # координатный спуск
        for k, (lo, hi) in space.items():
            for val in np.linspace(lo, hi, 13):
                q = {**best_p, k: val}
                v = objective(D, scorer(q), train_mask)
                if v > best_v + 1e-12:
                    best_p, best_v = q, v
    return best_p


def folds(D, k=5, seed=0):
    uniq = np.unique(D.groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    f = np.array([fold_of[g] for g in D.groups])
    return [(f != i, f == i) for i in range(k)]


def cv(D, scorer, seeds=(0, 1, 2), space=PARAM_SPACE, **kw):
    """Возвращает CV Acc@1 / MRR (на отложенных фолдах, агрегировано по всем фото)."""
    accs, mrrs = [], []
    for seed in seeds:
        full = np.zeros(D.X.shape[0])
        for train, test in folds(D, seed=seed):
            p = fit(D, scorer, train, space=space, **kw)
            s = scorer(p)
            sel = test[D.rec_idx]
            full[sel] = s[sel]
        a, m = metrics(D, full)
        accs.append(a)
        mrrs.append(m)
    return float(np.mean(accs)), float(np.mean(mrrs))
