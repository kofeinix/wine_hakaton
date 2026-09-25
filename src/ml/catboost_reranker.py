from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

VIEWS = ("original", "bottle_crop", "label_crop")
RRF_K = 60
VIEW_SCORE_TOP_K_PHOTOS = 3
VIEW_SCORE_BEST_WEIGHT = 0.75
VIEW_SCORE_MEAN_WEIGHT = 0.25
BASELINE_MAX_WEIGHT = 0.7
BASELINE_SUM_WEIGHT = 0.3
LABEL_PHOTO_AREA_THRESHOLD = 0.60
LABEL_PHOTO_CONFIDENCE_THRESHOLD = 0.50


@dataclass(frozen=True)
class CandidateFeatureRow:
    wine_id: str
    slug: str
    features: dict[str, float]
    baseline_score: float
    cosine_score: float
    max_score: float
    mean_score: float
    score_std: float
    n_photos: int
    view_photo_ids: dict[str, str]


class CatBoostWineReranker:
    def __init__(self, model_path: Path, feature_names: list[str]) -> None:
        try:
            from catboost import CatBoost
        except ImportError as exc:  # pragma: no cover - depends on optional package.
            raise RuntimeError(
                "catboost is not installed. Install project dependencies or run `uv add catboost`."
            ) from exc

        self.model_path = model_path
        self.feature_names = feature_names
        self.model = CatBoost()
        self.model.load_model(str(model_path))

    @classmethod
    def load(cls, model_path: str | Path, metadata_path: str | Path | None = None) -> "CatBoostWineReranker":
        model = Path(model_path)
        metadata = Path(metadata_path) if metadata_path else model.with_suffix(".json")
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        return cls(model, list(payload["feature_names"]))

    def predict_rows(self, rows: list[CandidateFeatureRow]) -> list[float]:
        matrix = [[row.features.get(name, 0.0) for name in self.feature_names] for row in rows]
        return [float(score) for score in self.model.predict(matrix)]


def rrf_score(rank: int | None, k: int = RRF_K) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / (k + rank)


def crop_features(crops: dict[str, Any] | None) -> dict[str, float]:
    if not crops:
        return {
            "label_crop_available": 0.0,
            "label_crop_confidence": 0.0,
            "label_crop_source_is_original": 0.0,
            "label_crop_source_is_bottle": 0.0,
            "label_crop_source_is_fallback": 0.0,
            "label_area_ratio": 0.0,
            "label_photo_mode": 0.0,
            "label_photo_area_threshold_delta": 0.0,
            "label_photo_confidence_threshold_delta": 0.0,
            "label_width_ratio": 0.0,
            "label_height_ratio": 0.0,
            "label_aspect": 0.0,
        }

    label = _as_crop_dict(crops.get("label_crop"))
    original = _as_crop_dict(crops.get("original"))
    label_available = bool(label.get("available")) if label else False
    label_width = float(label.get("width") or 0.0) if label else 0.0
    label_height = float(label.get("height") or 0.0) if label else 0.0
    original_width = float(original.get("width") or 0.0) if original else 0.0
    original_height = float(original.get("height") or 0.0) if original else 0.0
    area_ratio = 0.0
    width_ratio = 0.0
    height_ratio = 0.0

    box = label.get("box") if label else None
    source_view = label.get("source_view") if label else None
    if (
        box
        and source_view in {"original", "original_fallback"}
        and original_width > 0
        and original_height > 0
    ):
        x1, y1, x2, y2 = [float(value) for value in box]
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        area_ratio = (box_width * box_height) / max(1.0, original_width * original_height)
        width_ratio = box_width / original_width
        height_ratio = box_height / original_height

    return {
        "label_crop_available": 1.0 if label_available else 0.0,
        "label_crop_confidence": float(label.get("confidence") or 0.0) if label else 0.0,
        "label_crop_source_is_original": 1.0 if source_view == "original" else 0.0,
        "label_crop_source_is_bottle": 1.0 if source_view == "bottle_crop" else 0.0,
        "label_crop_source_is_fallback": 1.0 if source_view == "original_fallback" else 0.0,
        "label_area_ratio": area_ratio,
        "label_photo_mode": (
            1.0
            if area_ratio > LABEL_PHOTO_AREA_THRESHOLD
            and float(label.get("confidence") or 0.0) > LABEL_PHOTO_CONFIDENCE_THRESHOLD
            else 0.0
        ),
        "label_photo_area_threshold_delta": area_ratio - LABEL_PHOTO_AREA_THRESHOLD,
        "label_photo_confidence_threshold_delta": (
            (float(label.get("confidence") or 0.0) if label else 0.0)
            - LABEL_PHOTO_CONFIDENCE_THRESHOLD
        ),
        "label_width_ratio": width_ratio,
        "label_height_ratio": height_ratio,
        "label_aspect": label_width / label_height if label_height > 0 else 0.0,
    }


def build_candidate_feature_rows(
    wine_view_matches: dict[str, dict[str, list[Any]]],
    wine_slugs: dict[str, str] | None = None,
    weights: dict[str, float] | None = None,
    crops: dict[str, Any] | None = None,
    ocr_features: dict[str, dict[str, float]] | None = None,
) -> list[CandidateFeatureRow]:
    rows: list[CandidateFeatureRow] = []
    common_crop_features = crop_features(crops)
    wine_slugs = wine_slugs or {}
    weights = weights or {}
    ocr_features = ocr_features or {}

    for wine_id, views in wine_view_matches.items():
        features = dict(common_crop_features)
        weighted_scores: list[float] = []
        raw_best_scores: list[float] = []
        all_scores: list[float] = []
        best_rank_values: list[float] = []
        view_photo_ids: dict[str, str] = {}

        for view in VIEWS:
            matches = views.get(view, [])
            scores = [_match_float(match, "score") for match in matches]
            ranks = [_match_optional_int(match, "rank") for match in matches]
            best_idx = max(range(len(scores)), key=scores.__getitem__) if scores else None
            best_score = scores[best_idx] if best_idx is not None else 0.0
            top_scores = sorted(scores, reverse=True)[:VIEW_SCORE_TOP_K_PHOTOS]
            top_score_mean = mean(top_scores) if top_scores else 0.0
            view_score = (
                VIEW_SCORE_BEST_WEIGHT * best_score
                + VIEW_SCORE_MEAN_WEIGHT * top_score_mean
            ) if scores else 0.0
            best_rank = min((rank for rank in ranks if rank is not None), default=None)
            weighted = weights.get(view, 0.0) * view_score

            features[f"{view}_present"] = 1.0 if scores else 0.0
            features[f"{view}_weight"] = weights.get(view, 0.0)
            features[f"{view}_score_max"] = best_score
            features[f"{view}_score_mean"] = mean(scores) if scores else 0.0
            features[f"{view}_score_min"] = min(scores) if scores else 0.0
            features[f"{view}_score_std"] = pstdev(scores) if len(scores) > 1 else 0.0
            features[f"{view}_score_sum"] = sum(scores) if scores else 0.0
            features[f"{view}_score_top3_mean"] = top_score_mean
            features[f"{view}_score_top3_sum"] = sum(top_scores) if top_scores else 0.0
            features[f"{view}_view_score"] = view_score
            features[f"{view}_photo_count"] = float(len(scores))
            features[f"{view}_best_rank"] = float(best_rank or 0)
            features[f"{view}_best_rrf"] = rrf_score(best_rank)
            features[f"{view}_weighted_score"] = weighted

            if scores:
                weighted_scores.append(weighted)
                raw_best_scores.append(best_score)
                all_scores.extend(scores)
                if best_rank is not None:
                    best_rank_values.append(float(best_rank))
                if best_idx is not None:
                    photo_id = _match_str(matches[best_idx], "photo_id")
                    if photo_id:
                        view_photo_ids[view] = photo_id

        max_score = max(weighted_scores) if weighted_scores else 0.0
        mean_score = mean(weighted_scores) if weighted_scores else 0.0
        sum_score = sum(weighted_scores) if weighted_scores else 0.0
        baseline_score = (
            BASELINE_MAX_WEIGHT * max_score
            + BASELINE_SUM_WEIGHT * sum_score
        ) if weighted_scores else 0.0
        cosine_score = max(raw_best_scores) if raw_best_scores else 0.0
        score_std = pstdev(weighted_scores) if len(weighted_scores) > 1 else 0.0

        features["baseline_score"] = baseline_score
        features["baseline_max_score"] = max_score
        features["baseline_mean_score"] = mean_score
        features["baseline_sum_score"] = sum_score
        features["baseline_score_std"] = score_std
        features["best_cosine_score"] = cosine_score
        features["all_score_mean"] = mean(all_scores) if all_scores else 0.0
        features["all_score_std"] = pstdev(all_scores) if len(all_scores) > 1 else 0.0
        features["all_photo_count"] = float(len(all_scores))
        features["views_present_count"] = float(sum(1 for view in VIEWS if features[f"{view}_present"] > 0))
        features["best_rank_min"] = min(best_rank_values) if best_rank_values else 0.0
        features["best_rank_mean"] = mean(best_rank_values) if best_rank_values else 0.0
        features["label_minus_original"] = features["label_crop_score_max"] - features["original_score_max"]
        features["bottle_minus_original"] = features["bottle_crop_score_max"] - features["original_score_max"]
        features["label_minus_bottle"] = features["label_crop_score_max"] - features["bottle_crop_score_max"]
        features["label_original_product"] = features["label_crop_score_max"] * features["original_score_max"]
        features["label_bottle_product"] = features["label_crop_score_max"] * features["bottle_crop_score_max"]
        candidate_ocr_features = ocr_features.get(wine_id, {})
        features.update(_numeric_ocr_features(candidate_ocr_features))

        rows.append(
            CandidateFeatureRow(
                wine_id=wine_id,
                slug=wine_slugs.get(wine_id, wine_id),
                features={key: _finite(value) for key, value in features.items()},
                baseline_score=baseline_score,
                cosine_score=cosine_score,
                max_score=max_score,
                mean_score=mean_score,
                score_std=score_std,
                n_photos=len(all_scores),
                view_photo_ids=view_photo_ids,
            )
        )

    return rows


def feature_names_from_rows(rows: list[CandidateFeatureRow]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update(row.features)
    return sorted(names)


def _numeric_ocr_features(values: dict[str, Any]) -> dict[str, float]:
    names = {
        "ocr_applied",
        "ocr_score",
        "ocr_domain_score",
        "ocr_fuzzy_score",
        "ocr_tfidf_score",
        "ocr_grape_score",
        "ocr_grape_match_count",
        "ocr_grape_total_count",
        "ocr_grape_missing_count",
        "ocr_grape_coverage",
        "ocr_producer_score",
        "ocr_name_score",
        "ocr_color_score",
        "ocr_color_detected_count",
        "ocr_color_detected_match",
        "ocr_color_detected_mismatch",
        "ocr_sugar_score",
        "ocr_sugar_detected_count",
        "ocr_sugar_detected_match",
        "ocr_sugar_detected_mismatch",
        "ocr_wratio",
        "ocr_token_set_ratio",
        "ocr_wratio_score",
        "ocr_token_set_score",
        "ocr_text_length",
        "ocr_candidate_text_length",
    }
    return {name: _finite(float(values.get(name, 0.0) or 0.0)) for name in names}


def _as_crop_dict(crop: Any) -> dict[str, Any]:
    if crop is None:
        return {}
    if isinstance(crop, dict):
        return crop
    if hasattr(crop, "model_dump"):
        return crop.model_dump()
    return {
        "available": getattr(crop, "available", False),
        "box": getattr(crop, "box", None),
        "confidence": getattr(crop, "confidence", None),
        "width": getattr(crop, "width", None),
        "height": getattr(crop, "height", None),
    }


def _match_float(match: Any, field: str) -> float:
    if isinstance(match, dict):
        return float(match.get(field) or 0.0)
    return float(getattr(match, field, 0.0) or 0.0)


def _match_optional_int(match: Any, field: str) -> int | None:
    value = match.get(field) if isinstance(match, dict) else getattr(match, field, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _match_str(match: Any, field: str) -> str:
    if isinstance(match, dict):
        return str(match.get(field) or "")
    return str(getattr(match, field, "") or "")


def _finite(value: float) -> float:
    value = float(value)
    if math.isfinite(value):
        return value
    return 0.0
