"""Entity-based OCR matching.

Вместо сравнения всего OCR-текста с candidate_text через WRatio/token_set
сравниваем каждую сущность вина отдельно (producer, name, grapes, color, sugar).

Для каждой сущности строится несколько представлений:
  - нормализованный оригинал;
  - RU -> LAT транслитерация;
  - LAT -> RU транслитерация;
  - вариант без пробелов.

Поиск внутри OCR: exact substring -> fuzzy по скользящим окнам токенов
(для однословной сущности окно = одиночный токен). Возвращается лучший score 0..1.

Правила:
  - отсутствие сущности в OCR НЕ штрафуется: скор не ниже нейтрального
    (для producer/name/grape — max(fuzzy, NEUTRAL_SCORE), для color/sugar — NEUTRAL_SCORE);
  - явное противоречие (candidate red, OCR white) штрафуется;
  - у сущностей разные веса.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from statistics import mean

from rapidfuzz import fuzz
from transliterate import translit

from src.ml.text_normalization import normalize_match_text

logger = logging.getLogger(__name__)

# --- конфигурация -----------------------------------------------------------

ENTITY_WEIGHTS: dict[str, float] = {
    "grape": 0.35,
    "producer": 0.20,
    "name": 0.20,
    "color": 0.10,
    "sugar": 0.15,
}

NEUTRAL_SCORE = 0.5
MATCH_THRESHOLD = 0.80
CONTRADICTION_PENALTY = 0.25
MIN_TOKEN_LENGTH = 3


@dataclass(frozen=True)
class EntityMatch:
    """Результат сопоставления одной сущности с OCR-текстом."""

    entity: str
    score: float
    matched: bool
    contradiction: bool
    best_variant: str | None = None
    best_window: str | None = None
    variants: list[str] = field(default_factory=list)
    raw_score: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "entity": self.entity,
            "score": round(self.score, 4),
            "matched": self.matched,
            "contradiction": self.contradiction,
            "best_variant": self.best_variant,
            "best_window": self.best_window,
            "variants": self.variants,
            "raw_score": None if self.raw_score is None else round(self.raw_score, 4),
        }


@dataclass(frozen=True)
class OcrEntityScore:
    """Итоговый entity-скор по всем сущностям вина."""

    domain_score: float
    entities: dict[str, EntityMatch]

    def to_dict(self) -> dict[str, object]:
        return {
            "domain_score": round(self.domain_score, 4),
            "entities": {name: match.to_dict() for name, match in self.entities.items()},
        }

    @classmethod
    def zero(cls) -> OcrEntityScore:
        """Скор для кандидата без карточки вина: сравнивать нечего, бонуса нет."""
        return cls(
            domain_score=0.0,
            entities={
                name: EntityMatch(entity=name, score=0.0, matched=False, contradiction=False)
                for name in ENTITY_WEIGHTS
            },
        )


# --- представления сущности -------------------------------------------------

def _translit_ru_to_lat(value: str) -> str:
    try:
        return translit(value, "ru", reversed=True)
    except Exception:  # noqa: BLE001 - библиотека кидает на неоднозначных строках
        return value


def _translit_lat_to_ru(value: str) -> str:
    try:
        return translit(value, "ru")
    except Exception:  # noqa: BLE001
        return value


@lru_cache(maxsize=16384)
def _cached_representations(value: str) -> tuple[str, ...]:
    # Транслитерируем исходную строку, а не нормализованную: normalize_match_text
    # заменяет кириллические а/о/с/р/е на латиницу, и транслит смешанной строки
    # даёт мусор ("красное" -> "kpacnoe"). Апостроф из ь/ъ убираем, чтобы он не
    # разрезал слово на два токена ("совиньон" -> "sovin on").
    candidates = (
        value,
        _translit_ru_to_lat(value).replace("'", ""),
        _translit_lat_to_ru(value),
    )
    reps: set[str] = set()
    for candidate in candidates:
        normalized = normalize_match_text(candidate)
        if normalized:
            reps.add(normalized)
            reps.add(normalized.replace(" ", ""))
    return tuple(sorted(reps))


def entity_representations(value: str | None) -> list[str]:
    """Несколько представлений сущности для поиска в OCR."""
    if not value:
        return []
    return list(_cached_representations(value))


# --- поиск внутри OCR -------------------------------------------------------

def best_match_in_ocr(
    ocr_text: str,
    ocr_tokens: list[str],
    variants: list[str],
) -> tuple[float, str | None, str | None]:
    """Лучшее совпадение любой из variants внутри OCR.

    Возвращает (score 0..1, лучший вариант, лучшее окно OCR).
    """
    if not variants or not ocr_tokens:
        return 0.0, None, None

    padded = f" {ocr_text} "
    best_score = 0.0
    best_variant: str | None = None
    best_window: str | None = None

    for variant in variants:
        if not variant:
            continue
        # 1) точное вхождение подстроки
        if f" {variant} " in padded:
            return 1.0, variant, variant

        variant_tokens = variant.split()
        n = len(variant_tokens)

        # 2) скользящее окно по токенам OCR
        if n <= len(ocr_tokens):
            for i in range(len(ocr_tokens) - n + 1):
                window = " ".join(ocr_tokens[i : i + n])
                score = fuzz.ratio(variant, window) / 100.0
                if score > best_score:
                    best_score = score
                    best_variant = variant
                    best_window = window

    return best_score, best_variant, best_window


# --- скоринг сущностей ------------------------------------------------------

def _floor_unmatched(score: float) -> float:
    """Не нашли сущность в OCR — не штрафуем: опускаться ниже нейтрали нельзя.

    OCR часто читает только часть этикетки, поэтому низкий fuzzy-скор чаще
    означает "не прочитали", чем "другое вино".
    """
    return score if score >= MATCH_THRESHOLD else max(score, NEUTRAL_SCORE)


def _score_simple_entity(
    entity: str,
    values: list[str],
    ocr_text: str,
    ocr_tokens: list[str],
) -> EntityMatch:
    variants = [rep for value in values for rep in entity_representations(value)]
    if not variants:
        return EntityMatch(
            entity=entity,
            score=NEUTRAL_SCORE,
            matched=False,
            contradiction=False,
            variants=[],
        )
    score, best_variant, best_window = best_match_in_ocr(ocr_text, ocr_tokens, variants)
    return EntityMatch(
        entity=entity,
        score=_floor_unmatched(score),
        matched=score >= MATCH_THRESHOLD,
        contradiction=False,
        best_variant=best_variant,
        best_window=best_window,
        variants=variants,
        raw_score=score,
    )


def _score_grape_entity(
    grape_groups: list[list[str]],
    ocr_text: str,
    ocr_tokens: list[str],
) -> EntityMatch:
    if not grape_groups:
        return EntityMatch(
            entity="grape",
            score=NEUTRAL_SCORE,
            matched=False,
            contradiction=False,
            variants=[],
        )

    scores: list[float] = []
    all_variants: list[str] = []
    best_variant: str | None = None
    best_window: str | None = None
    best_score = 0.0
    for group in grape_groups:
        variants = [rep for value in group for rep in entity_representations(value)]
        all_variants.extend(variants)
        score, variant, window = best_match_in_ocr(ocr_text, ocr_tokens, variants)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_variant = variant
            best_window = window

    grape_score = mean(scores)
    return EntityMatch(
        entity="grape",
        score=_floor_unmatched(grape_score),
        matched=grape_score >= MATCH_THRESHOLD,
        contradiction=False,
        best_variant=best_variant,
        best_window=best_window,
        variants=sorted(set(all_variants)),
        raw_score=grape_score,
    )


def _score_categorical_entity(
    entity: str,
    candidate_key: str | None,
    candidate_variants: list[str],
    all_variants_by_key: dict[str, list[str]],
    ocr_text: str,
    ocr_tokens: list[str],
) -> EntityMatch:
    """Цвет/сахар: свой вариант ищем, чужой — считаем противоречием."""
    if not candidate_key:
        return EntityMatch(
            entity=entity,
            score=NEUTRAL_SCORE,
            matched=False,
            contradiction=False,
            variants=[],
        )

    own_variants = [rep for value in candidate_variants for rep in entity_representations(value)]
    own_score, own_variant, own_window = best_match_in_ocr(ocr_text, ocr_tokens, own_variants)

    other_best = 0.0
    other_variant: str | None = None
    for key, variants in all_variants_by_key.items():
        if key == candidate_key:
            continue
        reps = [rep for value in variants for rep in entity_representations(value)]
        score, variant, _ = best_match_in_ocr(ocr_text, ocr_tokens, reps)
        if score > other_best:
            other_best = score
            other_variant = variant

    if own_score >= MATCH_THRESHOLD:
        return EntityMatch(
            entity=entity,
            score=own_score,
            matched=True,
            contradiction=False,
            best_variant=own_variant,
            best_window=own_window,
            variants=own_variants,
            raw_score=own_score,
        )
    if other_best >= MATCH_THRESHOLD:
        return EntityMatch(
            entity=entity,
            score=max(0.0, own_score - CONTRADICTION_PENALTY),
            matched=False,
            contradiction=True,
            best_variant=other_variant,
            best_window=own_window,
            variants=own_variants,
            raw_score=own_score,
        )
    return EntityMatch(
        entity=entity,
        score=NEUTRAL_SCORE,
        matched=False,
        contradiction=False,
        best_variant=own_variant,
        best_window=own_window,
        variants=own_variants,
    )


def score_ocr_entities(
    ocr_text: str,
    *,
    name: str | None,
    producer: str | None,
    grape_groups: list[list[str]],
    color: str | None,
    color_aliases: dict[str, list[str]],
    sugar: str | None,
    sugar_variants: dict[str, list[str]],
) -> OcrEntityScore:
    """Считает entity-скор вина относительно OCR-текста."""
    ocr_tokens = ocr_text.split()

    entities: dict[str, EntityMatch] = {
        "grape": _score_grape_entity(grape_groups, ocr_text, ocr_tokens),
        "producer": _score_simple_entity(
            "producer", [producer] if producer else [], ocr_text, ocr_tokens
        ),
        "name": _score_simple_entity("name", [name] if name else [], ocr_text, ocr_tokens),
        "color": _score_categorical_entity(
            "color",
            normalize_match_text(color) or None,
            color_aliases.get(normalize_match_text(color), [color]) if color else [],
            color_aliases,
            ocr_text,
            ocr_tokens,
        ),
        "sugar": _score_categorical_entity(
            "sugar",
            normalize_match_text(sugar) or None,
            sugar_variants.get(normalize_match_text(sugar), [sugar]) if sugar else [],
            sugar_variants,
            ocr_text,
            ocr_tokens,
        ),
    }

    total_weight = sum(ENTITY_WEIGHTS.values())
    domain_score = (
        sum(ENTITY_WEIGHTS[key] * match.score for key, match in entities.items())
        / total_weight
    )
    return OcrEntityScore(domain_score=domain_score, entities=entities)