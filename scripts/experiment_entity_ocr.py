#!/usr/bin/env python3
"""Эксперимент: entity-based OCR matching вместо WRatio/token_set по всему тексту.

Гипотеза:
  - сравниваем не весь OCR с candidate_text, а каждую сущность отдельно
    (producer, name, grapes, color, sugar);
  - для сущности строим несколько представлений: оригинал, нормализация,
    транслитерация RU->LAT и LAT->RU;
  - ищем лучшее совпадение внутри OCR через exact match + fuzzy по словам
    и коротким окнам, возвращаем score 0..1;
  - отсутствие сущности в OCR НЕ штрафуем (нейтральное значение);
  - явное противоречие (candidate red, OCR white) штрафуем;
  - у сущностей разные веса.

Сравниваем с baseline (текущая формула) на data/ocr_visual_grid_responses.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services.wine_service import (  # noqa: E402
    OCR_DOMAIN_WEIGHT,
    OCR_FUZZY_TOKEN_SET_WEIGHT,
    OCR_FUZZY_WEIGHT,
    OCR_FUZZY_WRATIO_WEIGHT,
    SUGAR_VARIANTS,
    WineService,
)
from src.ml.text_normalization import normalize_match_text  # noqa: E402

RESPONSES = PROJECT_ROOT / "data" / "ocr_visual_grid_responses.json"
GRAPES = PROJECT_ROOT / "data" / "db" / "grapes.json"
GRAPE_ALIASES = PROJECT_ROOT / "data" / "db" / "grape_aliases.json"
COLOR_ALIASES = PROJECT_ROOT / "data" / "db" / "color_aliases.json"

# --- транслитерация ---------------------------------------------------------

RU_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
    "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y",
    "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
LAT_TO_RU = {
    "a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "e": "е", "z": "з",
    "i": "и", "y": "й", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о",
    "p": "п", "r": "р", "s": "с", "t": "т", "u": "у", "f": "ф", "h": "х",
    "c": "ц", "j": "дж", "w": "в", "x": "кс", "q": "к",
}


def translit_ru_to_lat(text: str) -> str:
    return "".join(RU_TO_LAT.get(ch, ch) for ch in text)


def translit_lat_to_ru(text: str) -> str:
    return "".join(LAT_TO_RU.get(ch, ch) for ch in text)


def representations(value: str) -> list[str]:
    """Несколько представлений сущности для поиска в OCR."""
    if not value:
        return []
    base = normalize_match_text(value)
    if not base:
        return []
    reps = {base}
    reps.add(normalize_match_text(translit_ru_to_lat(base)))
    reps.add(normalize_match_text(translit_lat_to_ru(base)))
    # вариант без пробелов (для слитных написаний)
    reps.add(base.replace(" ", ""))
    return [r for r in reps if r]


# --- поиск сущности в OCR ---------------------------------------------------

def best_match_in_ocr(ocr_text: str, ocr_tokens: list[str], variants: list[str]) -> float:
    """Лучшее совпадение любой из variants внутри OCR. 0..1."""
    if not variants or not ocr_tokens:
        return 0.0
    padded = f" {ocr_text} "
    best = 0.0
    for variant in variants:
        if not variant:
            continue
        # exact substring
        if f" {variant} " in padded:
            return 1.0
        vt = variant.split()
        n = len(vt)
        # скользящее окно по токенам OCR
        if n <= len(ocr_tokens):
            for i in range(len(ocr_tokens) - n + 1):
                window = " ".join(ocr_tokens[i:i + n])
                best = max(best, fuzz.ratio(variant, window) / 100.0)
        # одиночные токены (для коротких сущностей)
        if n == 1:
            for tok in ocr_tokens:
                best = max(best, fuzz.ratio(variant, tok) / 100.0)
    return best


# --- конфигурация сущностей -------------------------------------------------

ENTITY_WEIGHTS = {
    "grape": 0.35,
    "producer": 0.20,
    "name": 0.20,
    "color": 0.10,
    "sugar": 0.15,
}
NEUTRAL = 0.5          # отсутствие сущности в OCR
MATCH_THRESHOLD = 0.80  # выше — считаем, что сущность найдена
CONTRADICTION_PENALTY = 0.25


def load_grape_variants() -> dict[str, list[str]]:
    grapes = json.loads(GRAPES.read_text(encoding="utf-8"))
    aliases = json.loads(GRAPE_ALIASES.read_text(encoding="utf-8"))
    by_id: dict[str, list[str]] = {}
    for item in aliases:
        by_id.setdefault(item["grape_id"], []).append(item["alias"])
    result: dict[str, list[str]] = {}
    for grape in grapes:
        result[normalize_match_text(grape["name"])] = [grape["name"], *by_id.get(grape["id"], [])]
    return result


def load_color_aliases() -> dict[str, list[str]]:
    raw = json.loads(COLOR_ALIASES.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for item in raw:
        key = normalize_match_text(item["color"])
        result.setdefault(key, [item["color"]])
        result[key].append(item["alias"])
    return result


def entity_scores(
    ocr_text: str,
    wine: dict,
    grape_variants: dict[str, list[str]],
    color_aliases: dict[str, list[str]],
) -> dict[str, float]:
    ocr_tokens = ocr_text.split()

    # grape: средний best-match по сортам кандидата
    grape_names = wine.get("grapes") or []
    if grape_names:
        scores = []
        for name in grape_names:
            variants = grape_variants.get(normalize_match_text(name), [name])
            reps = [r for v in variants for r in representations(v)]
            scores.append(best_match_in_ocr(ocr_text, ocr_tokens, reps))
        grape_score = mean(scores)
    else:
        grape_score = NEUTRAL

    producer = wine.get("producer")
    producer_score = (
        best_match_in_ocr(ocr_text, ocr_tokens, representations(producer))
        if producer else NEUTRAL
    )

    name = wine.get("name")
    name_score = (
        best_match_in_ocr(ocr_text, ocr_tokens, representations(name))
        if name else NEUTRAL
    )

    # color: ищем свой цвет; если найден чужой — противоречие
    color = wine.get("color")
    if color:
        color_key = normalize_match_text(color)
        own_reps = [r for v in color_aliases.get(color_key, [color]) for r in representations(v)]
        own = best_match_in_ocr(ocr_text, ocr_tokens, own_reps)
        other_best = 0.0
        for key, variants in color_aliases.items():
            if key == color_key:
                continue
            reps = [r for v in variants for r in representations(v)]
            other_best = max(other_best, best_match_in_ocr(ocr_text, ocr_tokens, reps))
        if own >= MATCH_THRESHOLD:
            color_score = own
        elif other_best >= MATCH_THRESHOLD:
            color_score = max(0.0, own - CONTRADICTION_PENALTY)
        else:
            color_score = NEUTRAL
    else:
        color_score = NEUTRAL

    # sugar: аналогично
    sugar = wine.get("sugar")
    if sugar:
        sugar_key = normalize_match_text(sugar)
        own_reps = [r for v in SUGAR_VARIANTS.get(sugar_key, [sugar]) for r in representations(v)]
        own = best_match_in_ocr(ocr_text, ocr_tokens, own_reps)
        other_best = 0.0
        for key, variants in SUGAR_VARIANTS.items():
            if key == sugar_key:
                continue
            reps = [r for v in variants for r in representations(v)]
            other_best = max(other_best, best_match_in_ocr(ocr_text, ocr_tokens, reps))
        if own >= MATCH_THRESHOLD:
            sugar_score = own
        elif other_best >= MATCH_THRESHOLD:
            sugar_score = max(0.0, own - CONTRADICTION_PENALTY)
        else:
            sugar_score = NEUTRAL
    else:
        sugar_score = NEUTRAL

    total_w = sum(ENTITY_WEIGHTS.values())
    domain = (
        ENTITY_WEIGHTS["grape"] * grape_score
        + ENTITY_WEIGHTS["producer"] * producer_score
        + ENTITY_WEIGHTS["name"] * name_score
        + ENTITY_WEIGHTS["color"] * color_score
        + ENTITY_WEIGHTS["sugar"] * sugar_score
    ) / total_w
    return {
        "grape": grape_score,
        "producer": producer_score,
        "name": name_score,
        "color": color_score,
        "sugar": sugar_score,
        "domain": domain,
    }


def baseline_ocr_score(ocr: str, wine: dict, color_aliases: dict[str, list[str]]) -> float:
    from types import SimpleNamespace

    producer = SimpleNamespace(name=wine.get("producer") or "")
    grape_links = [
        SimpleNamespace(grape=SimpleNamespace(name=g, aliases=[]))
        for g in (wine.get("grapes") or [])
    ]
    wine_like = SimpleNamespace(
        producer=producer,
        name=wine.get("name") or "",
        year=wine.get("year"),
        color=wine.get("color"),
        sugar=wine.get("sugar"),
        grape_links=grape_links,
    )
    candidate_text = normalize_match_text(WineService._wine_to_ocr_candidate_text(wine_like))
    wratio = fuzz.WRatio(ocr, candidate_text) if candidate_text else 0.0
    token_set = fuzz.token_set_ratio(ocr, candidate_text) if candidate_text else 0.0
    fuzzy = (OCR_FUZZY_WRATIO_WEIGHT * wratio + OCR_FUZZY_TOKEN_SET_WEIGHT * token_set) / 100.0
    structured = WineService._structured_ocr_score(ocr, wine_like, color_aliases)
    return OCR_DOMAIN_WEIGHT * structured["domain_score"] + OCR_FUZZY_WEIGHT * fuzzy


def build_records(grape_variants, color_aliases) -> list[dict]:
    rows = json.loads(RESPONSES.read_text(encoding="utf-8"))
    records = []
    for row in rows:
        resp = row.get("response") or {}
        diag = resp.get("diagnostics") or {}
        ocr = str((diag.get("ocr_rerank") or {}).get("normalized_text") or "")
        matches = resp.get("result") or resp.get("results") or []
        candidates = []
        for match in matches:
            wine = match.get("wine") or {}
            if not wine:
                continue
            visual = float(match.get("max_score") or match.get("score") or 0.0)
            if ocr:
                base = baseline_ocr_score(ocr, wine, color_aliases)
                entity = entity_scores(ocr, wine, grape_variants, color_aliases)["domain"]
                hybrid = 0.5 * base + 0.5 * entity
            else:
                base = 0.0
                entity = 0.0
                hybrid = 0.0
            candidates.append(
                {
                    "wine_id": str(match["wine_id"]),
                    "visual": visual,
                    "base": base,
                    "entity": entity,
                    "hybrid": hybrid,
                }
            )
        records.append(
            {
                "expected": row["expected"],
                "image": row["image"],
                "ocr_applied": bool(ocr),
                "candidates": candidates,
            }
        )
    return records


def evaluate(records, score_key, visual_weight, ocr_weight) -> dict:
    ranks = []
    for record in records:
        cands = record["candidates"]
        if not cands:
            ranks.append(None)
            continue
        ranked = sorted(
            cands,
            key=lambda c: visual_weight * c["visual"] + ocr_weight * c[score_key],
            reverse=True,
        )
        rank = None
        for idx, cand in enumerate(ranked, start=1):
            if cand["wine_id"] == record["expected"]:
                rank = idx
                break
        ranks.append(rank)
    n = len(ranks)
    acc1 = sum(1 for r in ranks if r == 1) / n
    mrr = sum(1.0 / r for r in ranks if r) / n
    r5 = sum(1 for r in ranks if r and r <= 5) / n
    r10 = sum(1 for r in ranks if r and r <= 10) / n
    avg_rank = mean(r for r in ranks if r) if any(ranks) else None
    return {"acc1": acc1, "mrr": mrr, "r5": r5, "r10": r10, "avg_rank": avg_rank}


def main() -> None:
    grape_variants = load_grape_variants()
    color_aliases = load_color_aliases()
    records = build_records(grape_variants, color_aliases)
    print(f"records={len(records)}  ocr_applied={sum(1 for r in records if r['ocr_applied'])}")

    for score_key, title in (
        ("base", "A) BASELINE (domain + fuzzy WRatio/token_set)"),
        ("entity", "B) ENTITY-BASED (multi-repr, exact+fuzzy windows, contradiction penalty)"),
        ("hybrid", "C) HYBRID 0.5*baseline + 0.5*entity"),
    ):
        print(f"\n=== {title} ===")
        print(f"{'vis':>4} {'ocr':>4} {'Acc@1':>8} {'MRR':>8} {'R@5':>8} {'R@10':>8} {'avg_rank':>9}")
        best = None
        for i in range(11):
            vw = round(i / 10, 1)
            ow = round(1 - i / 10, 1)
            m = evaluate(records, score_key, vw, ow)
            print(
                f"{vw:>4.1f} {ow:>4.1f} {m['acc1']:>8.2%} {m['mrr']:>8.4f} "
                f"{m['r5']:>8.2%} {m['r10']:>8.2%} {m['avg_rank']:>9.3f}"
            )
            if best is None or m["acc1"] > best[0]:
                best = (m["acc1"], vw, ow, m)
        print(f"BEST: visual={best[1]} ocr={best[2]} Acc@1={best[0]:.2%} MRR={best[3]['mrr']:.4f}")


if __name__ == "__main__":
    main()