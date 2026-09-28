"""OCR-реранк кандидатов: сопоставление текста этикетки с карточками вин.

Итоговый скор кандидата:

    final = visual_score + ocr_bonus

ocr_bonus складывается только из бонусов за найденное и штрафов за явные
противоречия — отсутствие чего-либо в OCR не штрафуется (OCR часто читает
этикетку частично):

  * сходство названия / производителя / сортов (IDF-coverage, см. entity_coverage),
    превращённое в бонус выше порога ENTITY_BONUS_THRESHOLD;
  * "уникальные доказательства" — сумма редкостей (IDF по пулу кандидатов)
    найденных в OCR слов названия+производителя. Слово, общее для всей линейки
    ("Ркацители"), весит 0; слово, которое есть только у одного кандидата и
    найдено на этикетке ("десертное", "Belmas"), — много. Это главный сигнал,
    различающий вина одного производителя;
  * противоречия: на этикетке другой цвет / сахар / сорт / производитель;
  * совпадение крепости.

Все строки сравниваются в нескольких представлениях: как есть, RU->LAT и
LAT->RU транслитерация (и для сущностей, и для OCR).

Веса подобраны кросс-валидацией на 578 eval-фото:
Acc@1 86.2% (прежняя формула) -> 87.4% (CV), 88.2% на всём наборе.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache

from rapidfuzz import fuzz
from transliterate import translit

from src.ml.text_normalization import normalize_match_text

# --- веса формулы -------------------------------------------------------------

ENTITY_BONUS_WEIGHT = 0.10
NAME_WEIGHT = 0.50
PRODUCER_WEIGHT = 0.85
GRAPE_WEIGHT = 0.25
GRAPE_MEAN_WEIGHT = 0.5  # сорта: 0.5 * mean + 0.5 * max по сортам кандидата
ENTITY_BONUS_THRESHOLD = 0.55
UNIQUE_EVIDENCE_WEIGHT = 0.005
UNIQUE_EVIDENCE_MATCH_THRESHOLD = 0.78
COLOR_CONTRADICTION_PENALTY = 0.02
SUGAR_CONTRADICTION_PENALTY = 0.02
SUGAR_MATCH_BONUS = 0.01
GRAPE_CONTRADICTION_PENALTY = 0.03
PRODUCER_CONTRADICTION_PENALTY = 0.05
ALCOHOL_WEIGHT = 0.02

# --- параметры сопоставления --------------------------------------------------

COVERAGE_TOKEN_THRESHOLD = 0.75  # слово сущности считается найденным от этого сходства
DETECTION_THRESHOLD = 0.88  # сорт / производитель каталога "есть на этикетке"
CATEGORY_MATCH_THRESHOLD = 0.80  # цвет / сахар
ALCOHOL_MATCH_TOLERANCE = 0.3
ALCOHOL_MISMATCH_TOLERANCE = 1.0

_WORD_RE = re.compile(r"\w+")
_ALCOHOL_RE = re.compile(r"(\d{1,2}(?:[.,]\d{1,2})?)\s*%")
_NUMBER_RE = re.compile(r"\d{1,2}(?:[.,]\d{1,2})?")

# --- представления строк ------------------------------------------------------


def _translit_ru_to_lat(value: str) -> str:
    try:
        # апостроф из ь/ъ убираем, иначе он режет слово ("совиньон" -> "sovin on")
        return translit(value, "ru", reversed=True).replace("'", "")
    except Exception:  # noqa: BLE001 - библиотека кидает на неоднозначных строках
        return value


def _translit_lat_to_ru(value: str) -> str:
    try:
        return translit(value, "ru")
    except Exception:  # noqa: BLE001
        return value


@lru_cache(maxsize=65536)
def text_variants(value: str) -> tuple[str, ...]:
    """Нормализованный текст + RU->LAT и LAT->RU транслитерации.

    Транслитерируется исходная строка: normalize_match_text заменяет
    кириллические а/о/с/р/е на латиницу, и транслит смешанной строки — мусор.
    """
    out: list[str] = []
    for candidate in (value, _translit_ru_to_lat(value), _translit_lat_to_ru(value)):
        normalized = normalize_match_text(candidate)
        if normalized and normalized not in out:
            out.append(normalized)
    return tuple(out)


def _variants_of(values: Iterable[str | None]) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        if value:
            for variant in text_variants(value):
                if variant not in out:
                    out.append(variant)
    return tuple(out)


View = tuple[str, tuple[str, ...]]  # (нормализованный текст, токены)


def ocr_views(raw_text: str) -> tuple[View, ...]:
    return tuple((variant, tuple(variant.split())) for variant in text_variants(raw_text))


def _words(value: str | None) -> list[str]:
    return [word for word in _WORD_RE.findall(value or "") if normalize_match_text(word)]


# --- сопоставление ------------------------------------------------------------


def window_match(variants: tuple[str, ...], views: tuple[View, ...]) -> float:
    """Лучшее сходство фразы с окном OCR той же длины (1.0 при точном вхождении)."""
    best = 0.0
    for text, tokens in views:
        padded = f" {text} "
        for variant in variants:
            if f" {variant} " in padded:
                return 1.0
            n = len(variant.split())
            for i in range(len(tokens) - n + 1):
                best = max(best, fuzz.ratio(variant, " ".join(tokens[i : i + n])) / 100.0)
            nospace = variant.replace(" ", "")
            if nospace != variant:
                for size in range(1, n + 1):
                    for i in range(len(tokens) - size + 1):
                        best = max(best, fuzz.ratio(nospace, "".join(tokens[i : i + size])) / 100.0)
    return best


def _token_best(token: str, tokens: tuple[str, ...]) -> float:
    """Лучшее сходство слова с одним словом OCR или склейкой двух соседних."""
    best = 0.0
    for i, other in enumerate(tokens):
        best = max(best, fuzz.ratio(token, other))
        if i + 1 < len(tokens):
            best = max(best, fuzz.ratio(token, other + tokens[i + 1]))
    return best / 100.0


@dataclass(frozen=True)
class TokenIdf:
    df: dict[str, int]
    n: int

    @classmethod
    def build(cls, values: Iterable[str]) -> TokenIdf:
        df: Counter = Counter()
        n = 0
        for value in values:
            n += 1
            df.update({token for variant in text_variants(value) for token in variant.split()})
        return cls(df=dict(df), n=n)

    def __call__(self, token: str) -> float:
        return math.log((self.n + 1) / (self.df.get(token, 0) + 1)) + 1.0


def entity_coverage(variants: tuple[str, ...], views: tuple[View, ...], idf: TokenIdf) -> float:
    """Какая доля сущности есть в OCR: слова взвешены по idf(слово) * длина.

    Порядок слов не важен ("Семильон - Алиготе" ~ "алиготе семильон").
    """
    best = 0.0
    for _, tokens in views:
        for variant in variants:
            words = [word for word in variant.split() if len(word) >= 2]
            num = den = 0.0
            for word in words:
                weight = idf(word) * len(word)
                score = _token_best(word, tokens)
                num += weight * (score if score >= COVERAGE_TOKEN_THRESHOLD else 0.0)
                den += weight
            if den:
                best = max(best, num / den)
    return best


def _bonus(score: float) -> float:
    if score < 0:
        return 0.0
    return min(1.0, max(0.0, (score - ENTITY_BONUS_THRESHOLD) / (1.0 - ENTITY_BONUS_THRESHOLD)))


# --- каталог и запрос -------------------------------------------------------------


@dataclass(frozen=True)
class OcrCatalog:
    """Словарь каталога: IDF слов и справочники для поиска противоречий."""

    name_idf: TokenIdf
    producer_idf: TokenIdf
    grape_idf: TokenIdf
    grapes: dict[str, tuple[str, ...]]  # ключ сорта -> название и синонимы
    producers: dict[str, tuple[str, ...]]  # название производителя -> название и синонимы
    color_variants: dict[str, list[str]]
    sugar_variants: dict[str, list[str]]

    @classmethod
    def build(
        cls,
        *,
        wine_names: Iterable[str],
        producer_names: Iterable[str],
        grapes: dict[str, list[str]],
        color_variants: dict[str, list[str]],
        sugar_variants: dict[str, list[str]],
        producer_aliases: dict[str, list[str]] | None = None,
    ) -> OcrCatalog:
        aliases = producer_aliases or {}
        producers = {
            name: tuple(dict.fromkeys([name, *aliases.get(name, [])]))
            for name in dict.fromkeys(name for name in producer_names if name)
        }
        return cls(
            name_idf=TokenIdf.build(name for name in wine_names if name),
            producer_idf=TokenIdf.build(spelling for spellings in producers.values() for spelling in spellings),
            grape_idf=TokenIdf.build(name for names in grapes.values() for name in names),
            grapes={key: tuple(names) for key, names in grapes.items()},
            producers=producers,
            color_variants=color_variants,
            sugar_variants=sugar_variants,
        )


@dataclass(frozen=True)
class OcrQuery:
    """OCR-текст одного запроса и то, что в нём найдено по всему каталогу."""

    raw_text: str
    views: tuple[View, ...]
    detected_grapes: frozenset[str]
    detected_producers: frozenset[str]
    color_scores: dict[str, float]
    sugar_scores: dict[str, float]
    alcohol_values: tuple[float, ...]

    @classmethod
    def build(cls, raw_text: str, catalog: OcrCatalog) -> OcrQuery:
        views = ocr_views(raw_text)
        return cls(
            raw_text=raw_text,
            views=views,
            detected_grapes=frozenset(
                key
                for key, names in catalog.grapes.items()
                if window_match(_variants_of(names), views) >= DETECTION_THRESHOLD
            ),
            detected_producers=frozenset(
                normalize_match_text(name)
                for name, spellings in catalog.producers.items()
                if window_match(_variants_of(spellings), views) >= DETECTION_THRESHOLD
            ),
            color_scores={
                key: window_match(_variants_of(values), views)
                for key, values in catalog.color_variants.items()
            },
            sugar_scores={
                key: window_match(_variants_of(values), views)
                for key, values in catalog.sugar_variants.items()
            },
            alcohol_values=_alcohol_numbers(raw_text),
        )


@dataclass(frozen=True)
class OcrCandidate:
    wine_id: str
    name: str | None
    producer: str | None
    grapes: dict[str, tuple[str, ...]]  # ключ сорта -> название и синонимы
    color: str | None
    sugar: str | None
    alcohol: str | None
    producer_aliases: tuple[str, ...] = ()  # написания производителя как на этикетке («Chateau Pinot»)


@dataclass(frozen=True)
class OcrCandidateScore:
    wine_id: str
    bonus: float  # добавка к визуальному скору
    features: dict[str, float] = field(default_factory=dict)

    def diagnostics(self) -> dict[str, float]:
        return {"wine_id": self.wine_id, "ocr_bonus": round(self.bonus, 4)} | {
            key.removeprefix("ocr_"): round(value, 4)
            for key, value in self.features.items()
            if key not in ("ocr_applied", "ocr_score")
        }


# --- скоринг --------------------------------------------------------------------


def _category(scores: dict[str, float], value: str | None) -> tuple[float, float, bool, bool]:
    """(own, other, match, contradiction) для цвета/сахара; -1 = значения нет."""
    key = normalize_match_text(value) if value else ""
    own = scores.get(key, -1.0) if key else -1.0
    other = max((score for k, score in scores.items() if k != key), default=0.0)
    match = own >= CATEGORY_MATCH_THRESHOLD
    contradiction = 0.0 <= own < CATEGORY_MATCH_THRESHOLD and other >= CATEGORY_MATCH_THRESHOLD
    return own, other, match, contradiction


def _alcohol_numbers(value: object) -> tuple[float, ...]:
    if not value:
        return ()
    if isinstance(value, int | float):
        number = float(value)
        return (number,) if 5.0 <= number <= 25.0 else ()
    text = str(value)
    if "%" not in text:
        return ()
    result: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        number = float(match.group(0).replace(",", "."))
        if 5.0 <= number <= 25.0:
            result.append(number)
    return tuple(result)


def _unique_evidence(query: OcrQuery, candidates: list[OcrCandidate]) -> list[tuple[float, float]]:
    """Для каждого кандидата: (сумма IDF_pool найденных слов, доля от его слов)."""
    word_score: dict[str, float] = {}

    def score(word: str) -> float:
        if word not in word_score:
            word_score[word] = max(
                (_token_best(variant.replace(" ", ""), tokens)
                 for variant in text_variants(word) for _, tokens in query.views),
                default=0.0,
            )
        return word_score[word]

    docs: list[dict[str, float]] = []
    for candidate in candidates:
        doc: dict[str, float] = {}
        for word in _words(candidate.name) + _words(candidate.producer):
            key = normalize_match_text(word)
            doc[key] = max(doc.get(key, 0.0), score(word))
        docs.append(doc)
    df: Counter = Counter(key for doc in docs for key in doc)
    n = len(docs)
    result = []
    for doc in docs:
        found = total = 0.0
        for key, value in doc.items():
            if key.isdigit():  # год/объём в названии — шум, а не отличие
                continue
            idf = math.log((n - df[key] + 0.5) / (df[key] + 0.5) + 1.0)
            total += idf
            if value >= UNIQUE_EVIDENCE_MATCH_THRESHOLD:
                found += idf
        result.append((found, found / total if total else 0.0))
    return result


def score_candidates(
    query: OcrQuery,
    catalog: OcrCatalog,
    candidates: list[OcrCandidate],
) -> list[OcrCandidateScore]:
    """OCR-бонус для пула кандидатов (IDF "уникальных доказательств" считается по пулу)."""
    evidence = _unique_evidence(query, candidates)
    scores: list[OcrCandidateScore] = []
    for candidate, (unique, unique_share) in zip(candidates, evidence, strict=True):
        views = query.views
        name = (
            entity_coverage(text_variants(candidate.name), views, catalog.name_idf)
            if candidate.name else -1.0
        )
        producer = (
            entity_coverage(_variants_of((candidate.producer, *candidate.producer_aliases)), views, catalog.producer_idf)
            if candidate.producer else -1.0
        )
        grape_scores = [
            entity_coverage(_variants_of(names), views, catalog.grape_idf)
            for names in candidate.grapes.values()
        ]
        grape_mean = sum(grape_scores) / len(grape_scores) if grape_scores else -1.0
        grape_max = max(grape_scores) if grape_scores else -1.0
        grape = GRAPE_MEAN_WEIGHT * grape_mean + (1 - GRAPE_MEAN_WEIGHT) * grape_max if grape_scores else -1.0

        color_own, color_other, _, color_contra = _category(query.color_scores, candidate.color)
        sugar_own, sugar_other, sugar_match, sugar_contra = _category(query.sugar_scores, candidate.sugar)

        own_grapes = set(candidate.grapes)
        grape_hit = len(query.detected_grapes & own_grapes)
        grape_extra = len(query.detected_grapes - own_grapes)
        grape_contra = bool(own_grapes) and grape_extra > 0 and grape_hit == 0

        producer_key = normalize_match_text(candidate.producer) if candidate.producer else ""
        # производитель с этикетки упомянут в названии вина («Golubitskoe Estate Chardonnay») — это совпадение
        name_key = f" {normalize_match_text(candidate.name)} " if candidate.name else ""
        producer_in_name = any(f" {detected} " in name_key for detected in query.detected_producers)
        producer_detected = producer_key in query.detected_producers or producer_in_name
        # «другой производитель» — только если производитель вина известен и на этикетке его нет
        producer_contra = (
            bool(producer_key) and bool(query.detected_producers - {producer_key}) and not producer_detected
        )

        alcohol_match = alcohol_miss = False
        candidate_alcohol_values = _alcohol_numbers(candidate.alcohol)
        if candidate_alcohol_values and query.alcohol_values:
            delta = min(
                abs(query_value - candidate_value)
                for query_value in query.alcohol_values
                for candidate_value in candidate_alcohol_values
            )
            alcohol_match = delta <= ALCOHOL_MATCH_TOLERANCE
            alcohol_miss = delta > ALCOHOL_MISMATCH_TOLERANCE

        bonus = (
            ENTITY_BONUS_WEIGHT
            * (NAME_WEIGHT * _bonus(name) + PRODUCER_WEIGHT * _bonus(producer) + GRAPE_WEIGHT * _bonus(grape))
            + UNIQUE_EVIDENCE_WEIGHT * unique
            - COLOR_CONTRADICTION_PENALTY * color_contra
            - SUGAR_CONTRADICTION_PENALTY * sugar_contra
            + SUGAR_MATCH_BONUS * sugar_match
            - GRAPE_CONTRADICTION_PENALTY * grape_contra
            - PRODUCER_CONTRADICTION_PENALTY * producer_contra
            + ALCOHOL_WEIGHT * (alcohol_match - alcohol_miss)
        )
        features = {
            "ocr_applied": 1.0,
            "ocr_score": bonus,
            "ocr_name_coverage": name,
            "ocr_producer_coverage": producer,
            "ocr_grape_coverage_mean": grape_mean,
            "ocr_grape_coverage_max": grape_max,
            "ocr_name_window": window_match(text_variants(candidate.name), views) if candidate.name else -1.0,
            "ocr_producer_window": (
                window_match(_variants_of((candidate.producer, *candidate.producer_aliases)), views)
                if candidate.producer
                else -1.0
            ),
            "ocr_unique_evidence": unique,
            "ocr_unique_evidence_share": unique_share,
            "ocr_color_own": color_own,
            "ocr_color_other": color_other,
            "ocr_color_contradiction": float(color_contra),
            "ocr_sugar_own": sugar_own,
            "ocr_sugar_other": sugar_other,
            "ocr_sugar_contradiction": float(sugar_contra),
            "ocr_grape_detected": float(len(query.detected_grapes)),
            "ocr_grape_hit": float(grape_hit),
            "ocr_grape_extra": float(grape_extra),
            "ocr_grape_contradiction": float(grape_contra),
            "ocr_producer_detected": float(producer_detected),
            "ocr_producer_contradiction": float(producer_contra),
            "ocr_alcohol_match": float(alcohol_match),
            "ocr_alcohol_mismatch": float(alcohol_miss),
            "ocr_text_tokens": float(len(query.views[0][1]) if query.views else 0),
        }
        scores.append(OcrCandidateScore(wine_id=candidate.wine_id, bonus=float(bonus), features=features))
    return scores
