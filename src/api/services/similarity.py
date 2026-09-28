"""Сходство для интерфейса: итоговый скор, нормированный на максимально возможный в этом поиске.

Итоговый скор = визуальный + бонус OCR. Его потолок зависит от ракурсов, по которым шёл поиск
(веса нормируются по участвующим ракурсам), поэтому «сырое» значение для одного и того же вина
было 0,76 по всему фото и 1,04 по крупной этикетке. Делим на потолок — получаем 0–100%.

Внутри одного поиска потолок общий для всех кандидатов, так что порядок вин не меняется.
"""

from __future__ import annotations

from src.api.services.visual_search import (
    VIEW_SCORE_BEST_WEIGHT,
    VIEW_SCORE_MEAN_WEIGHT,
)
from src.ml.ocr_matching import (
    ALCOHOL_WEIGHT,
    ENTITY_BONUS_WEIGHT,
    GRAPE_WEIGHT,
    NAME_WEIGHT,
    PRODUCER_WEIGHT,
    SUGAR_MATCH_BONUS,
)

# полное подтверждение текстом: название, производитель и сорта найдены целиком, совпали сахар
# и крепость (≈0,19). Слагаемое «уникальные слова» потолка не имеет — выше него просто 100%
OCR_FULL_BONUS = (
    ENTITY_BONUS_WEIGHT * (NAME_WEIGHT + PRODUCER_WEIGHT + GRAPE_WEIGHT) + SUGAR_MATCH_BONUS + ALCOHOL_WEIGHT
)


def max_visual_score(weights: dict[str, float]) -> float:
    """Визуальный скор при идеальном совпадении всех ракурсов (косинус 1): 0,7·max(w) + 0,3·Σw."""
    if not weights:
        return 0.0
    perfect_view = VIEW_SCORE_BEST_WEIGHT + VIEW_SCORE_MEAN_WEIGHT  # = 1
    return 0.7 * max(weights.values()) * perfect_view + 0.3 * sum(weights.values()) * perfect_view


def max_score(weights: dict[str, float], ocr_applied: bool) -> float:
    """Потолок итогового скора; без OCR бонуса нет — и в потолке его не учитываем."""
    return max_visual_score(weights) + (OCR_FULL_BONUS if ocr_applied else 0.0)


def similarities(scores: list[float], weights: dict[str, float], ocr_applied: bool) -> list[float]:
    ceiling = max_score(weights, ocr_applied)
    if ceiling <= 0:
        return [0.0 for _ in scores]
    return [min(max(score / ceiling, 0.0), 1.0) for score in scores]
