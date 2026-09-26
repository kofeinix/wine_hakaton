"""Токен-уровень: для каждого слова карточки кандидата — насколько похожее слово есть в OCR.

Сохраняет для каждой записи список кандидатов, у каждого — [(field, token_key, score)].
token_key — нормализованное исходное слово (для подсчёта редкости среди кандидатов).
"""

from __future__ import annotations

import pickle
import re
from multiprocessing import Pool

from rapidfuzz import fuzz

from lab_data import lab_cache, load_records, norm
from lab_features import CAT, variants, views

OUT = lab_cache() / "tokens.pkl"
WORD_RE = re.compile(r"\w+")


def _words(value: str | None) -> list[str]:
    return [w for w in WORD_RE.findall(value or "") if norm(w)]


def _best(word: str, vw) -> float:
    best = 0.0
    for v in variants(word):
        v = v.replace(" ", "")
        for text, toks in vw:
            for i, o in enumerate(toks):
                s = fuzz.ratio(v, o)
                if i + 1 < len(toks):
                    s = max(s, fuzz.ratio(v, o + toks[i + 1]))
                if s > best:
                    best = s
                    if best == 100:
                        return 1.0
    return best / 100.0


def record_tokens(rec: dict):
    out = {}
    for tr in (True, False):
        vw = views(rec["raw_ocr"], tr)
        cache: dict[str, float] = {}

        def score(word: str) -> float:
            if word not in cache:
                cache[word] = _best(word, vw)
            return cache[word]

        cands = []
        for cand in rec["cands"]:
            w = CAT["wines"][cand["wine_id"]]
            toks = []
            seen = set()
            for field, value in (("n", w["name"]), ("p", w["producer"])):
                for word in _words(value):
                    key = (field, norm(word))
                    if key not in seen:
                        seen.add(key)
                        toks.append((field, norm(word), score(word)))
            for gid in w["grape_ids"]:
                # сорт — одно "слово" группы: лучший из синонимов
                names = CAT["grapes"][gid]
                s = max(max((score(x) for x in _words(n)), default=0.0) if len(_words(n)) == 1
                        else _multi(n, score) for n in names)
                toks.append(("g", gid, s))
            cands.append(toks)
        out[tr] = cands
    return out


def _multi(name: str, score) -> float:
    ws = _words(name)
    return sum(score(x) for x in ws) / len(ws) if ws else 0.0


if __name__ == "__main__":
    import time

    t = time.time()
    with Pool(12) as pool:
        res = pool.map(record_tokens, load_records(), chunksize=4)
    OUT.write_bytes(pickle.dumps(res))
    print(f"done {time.time() - t:.0f}s")
