"""Признаки OCR-матчинга для всех пар (фото, кандидат) по нескольким алгоритмам."""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool

from rapidfuzz import fuzz
from transliterate import translit

from lab_data import lab_cache, load_catalog, load_records, norm

from src.api.services.wine_service import SUGAR_VARIANTS  # noqa: E402

OUT = lab_cache() / "features.pkl"
DETECT_THRESHOLD = 0.88

# ---------------------------------------------------------------- представления


def _ru2lat(s: str) -> str:
    try:
        return translit(s, "ru", reversed=True).replace("'", "")
    except Exception:
        return s


def _lat2ru(s: str) -> str:
    try:
        return translit(s, "ru")
    except Exception:
        return s


@lru_cache(maxsize=200_000)
def variants(value: str) -> tuple[str, ...]:
    out = []
    for cand in (value, _ru2lat(value), _lat2ru(value)):
        n = norm(cand)
        if n and n not in out:
            out.append(n)
    return tuple(out)


@lru_cache(maxsize=4096)
def views(raw: str, translit_ocr: bool) -> tuple[tuple[str, tuple[str, ...]], ...]:
    cands = (raw, _ru2lat(raw), _lat2ru(raw)) if translit_ocr else (raw,)
    out = []
    for cand in cands:
        n = norm(cand)
        if n and n not in [v for v, _ in out]:
            out.append((n, tuple(n.split())))
    return tuple(out)


# ---------------------------------------------------------------- матчеры


def m_win(vs, vw, spread: int = 0) -> float:
    best = 0.0
    for text, toks in vw:
        padded = f" {text} "
        for v in vs:
            if f" {v} " in padded:
                return 1.0
            n = len(v.split())
            for size in range(max(1, n - spread), n + spread + 1):
                for i in range(len(toks) - size + 1):
                    s = fuzz.ratio(v, " ".join(toks[i : i + size])) / 100.0
                    if s > best:
                        best = s
            nospace = v.replace(" ", "")
            if nospace != v:
                for size in range(1, n + 1):
                    for i in range(len(toks) - size + 1):
                        s = fuzz.ratio(nospace, "".join(toks[i : i + size])) / 100.0
                        if s > best:
                            best = s
    return best


def _token_best(t: str, toks: tuple[str, ...]) -> float:
    best = 0.0
    for i, o in enumerate(toks):
        s = fuzz.ratio(t, o)
        if s > best:
            best = s
        if i + 1 < len(toks):  # слово, разорванное OCR
            s = fuzz.ratio(t, o + toks[i + 1])
            if s > best:
                best = s
    return best / 100.0


def m_cov(vs, vw, tau: float, weight) -> float:
    """Доля (взвешенная) токенов сущности, найденных в OCR."""
    best = 0.0
    for text, toks in vw:
        for v in vs:
            tokens = [t for t in v.split() if len(t) >= 2]
            if not tokens:
                continue
            num = den = 0.0
            for t in tokens:
                w = weight(t)
                s = _token_best(t, toks)
                num += w * (s if s >= tau else 0.0)
                den += w
            if den and num / den > best:
                best = num / den
    return best


def m_partial(vs, vw) -> float:
    best = 0.0
    for text, _ in vw:
        for v in vs:
            if len(v) < 4:
                continue
            s = fuzz.partial_ratio(v, text) / 100.0
            if s > best:
                best = s
    return best


# ---------------------------------------------------------------- IDF по каталогу

CAT = load_catalog()


def _build_idf(values: list[str]):
    df: Counter = Counter()
    for value in values:
        toks = {t for v in variants(value) for t in v.split()}
        df.update(toks)
    n = len(values)

    def idf(t: str) -> float:
        return math.log((n + 1) / (df.get(t, 0) + 1)) + 1.0

    return idf


IDF = {
    "name": _build_idf([w["name"] for w in CAT["wines"].values()]),
    "producer": _build_idf(list(CAT["producers"].values())),
    "grape": _build_idf([n for names in CAT["grapes"].values() for n in names]),
}


def matchers(entity: str):
    idf = IDF[entity]
    return {
        "win": lambda vs, vw: m_win(vs, vw),
        "win1": lambda vs, vw: m_win(vs, vw, spread=1),
        "cov": lambda vs, vw: m_cov(vs, vw, 0.0, len),
        "covt": lambda vs, vw: m_cov(vs, vw, 0.75, len),
        "idf": lambda vs, vw: m_cov(vs, vw, 0.75, idf),
        "idfl": lambda vs, vw: m_cov(vs, vw, 0.75, lambda t: idf(t) * len(t)),
        "part": lambda vs, vw: m_partial(vs, vw),
    }


MATCHERS = {e: matchers(e) for e in ("name", "producer", "grape")}

# ---------------------------------------------------------------- признаки записи

ALC_RE = re.compile(r"(\d{1,2}(?:[.,]\d{1,2})?)\s*%")
NUMBER_RE = re.compile(r"\d{1,2}(?:[.,]\d{1,2})?")


def alcohol_numbers(value: object) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, int | float):
        number = float(value)
        return (number,) if 5.0 <= number <= 25.0 else ()
    text = str(value)
    if "%" not in text:
        return ()
    result: list[float] = []
    for match in NUMBER_RE.finditer(text):
        try:
            number = float(match.group(0).replace(",", "."))
        except ValueError:
            continue
        if 5.0 <= number <= 25.0:
            result.append(number)
    return tuple(result)


def _vs_of(values) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        if value:
            for v in variants(value):
                if v not in out:
                    out.append(v)
    return tuple(out)


def record_features(rec: dict) -> list[dict]:
    raw = rec["raw_ocr"]
    feats_all = []
    vw_by = {tr: views(raw, tr) for tr in (False, True)}
    vw = vw_by[True]

    # детекция по каталогу: какие сорта / производители / цвета / сахар есть в OCR
    detected_grapes = {
        gid for gid, names in CAT["grapes"].items() if m_win(_vs_of(names), vw) >= DETECT_THRESHOLD
    }
    detected_producers = {
        pid for pid, name in CAT["producers"].items() if m_win(_vs_of([name]), vw) >= DETECT_THRESHOLD
    }
    color_scores = {k: m_win(_vs_of(vals), vw) for k, vals in CAT["color_aliases"].items()}
    sugar_scores = {k: m_win(_vs_of(vals), vw) for k, vals in SUGAR_VARIANTS.items()}
    alc_values = alcohol_numbers(raw)

    entity_cache: dict = {}

    def ent(entity: str, values: tuple, tr: bool) -> dict:
        key = (entity, values, tr)
        if key not in entity_cache:
            vs = _vs_of(values)
            entity_cache[key] = (
                {name: fn(vs, vw_by[tr]) for name, fn in MATCHERS[entity].items()} if vs else None
            )
        return entity_cache[key]

    for cand in rec["cands"]:
        wine = CAT["wines"][cand["wine_id"]]
        f: dict[str, float] = {"visual": cand["visual"]}
        for tr in (False, True):
            suffix = "" if tr else "_notr"
            for entity, values in (("name", (wine["name"],)), ("producer", (wine["producer"],))):
                res = ent(entity, values, tr)
                for mname in MATCHERS[entity]:
                    f[f"{entity}_{mname}{suffix}"] = res[mname] if res else -1.0
            groups = [tuple(CAT["grapes"][gid]) for gid in wine["grape_ids"]]
            per_group = [ent("grape", g, tr) for g in groups]
            for mname in MATCHERS["grape"]:
                scores = [g[mname] for g in per_group if g]
                f[f"grape_{mname}_mean{suffix}"] = sum(scores) / len(scores) if scores else -1.0
                f[f"grape_{mname}_max{suffix}"] = max(scores) if scores else -1.0
        # противоречия/подтверждения по каталогу
        cand_grapes = set(wine["grape_ids"])
        f["g_n"] = float(len(cand_grapes))
        f["g_det"] = float(len(detected_grapes))
        f["g_hit"] = float(len(detected_grapes & cand_grapes))
        f["g_extra"] = float(len(detected_grapes - cand_grapes))
        pid = next((k for k, v in CAT["producers"].items() if v == wine["producer"]), None)
        f["p_hit"] = float(pid in detected_producers)
        f["p_other"] = float(bool(detected_producers - {pid}) and pid not in detected_producers)
        ck = norm(wine["color"]) if wine["color"] else None
        own_c = color_scores.get(ck, -1.0) if ck else -1.0
        other_c = max((s for k, s in color_scores.items() if k != ck), default=0.0)
        f["c_own"], f["c_other"] = own_c, other_c
        sk = norm(wine["sugar"]) if wine["sugar"] else None
        own_s = sugar_scores.get(sk, -1.0) if sk else -1.0
        other_s = max((s for k, s in sugar_scores.items() if k != sk), default=0.0)
        f["s_own"], f["s_other"] = own_s, other_s
        wa = alcohol_numbers(wine["alcohol"])
        delta = min((abs(a - candidate_a) for a in alc_values for candidate_a in wa), default=None)
        f["alc_match"] = float(delta is not None and delta <= 0.3)
        f["alc_miss"] = float(delta is not None and delta > 1.0)
        f["ocr_len"] = float(len(rec["ocr"].split()))
        feats_all.append(f)
    return feats_all


if __name__ == "__main__":
    import time

    records = load_records()
    t0 = time.time()
    with Pool(8) as pool:
        feats = pool.map(record_features, records, chunksize=4)
    OUT.write_bytes(pickle.dumps(feats))
    print(f"done in {time.time() - t0:.0f}s; features per cand: {len(feats[0][0])}")
