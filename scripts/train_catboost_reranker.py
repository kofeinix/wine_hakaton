#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services.photo_service import WinePhotoService
from src.api.repositories.wine_repository import WineRepository
from src.api.services.wine_service import (
    GLOBAL_CANDIDATE_LIMIT,
    LABEL_PHOTO_AREA_THRESHOLD,
    LABEL_PHOTO_CONFIDENCE_THRESHOLD,
    OCR_DOMAIN_WEIGHT,
    OCR_FUZZY_TOKEN_SET_WEIGHT,
    OCR_FUZZY_WEIGHT,
    OCR_FUZZY_WRATIO_WEIGHT,
    PER_VIEW_TOP_K,
    VIEW_WEIGHTS,
    WineService,
)
from src.connections.database.postgres import DatabaseClient
from src.connections.qdrant import QdrantClient
from src.llm.langchain_openai import ChatOpenAIWrapper
from src.ml.catboost_reranker import (
    build_candidate_feature_rows,
)
from src.ml.siglip2 import SiglipImageEmbedder
from src.ml.yolo import YoloBottleCropper, YoloLabelCropper
from src.settings.settings import YoloSettings, all_settings


EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "catboost_reranker_dataset.jsonl"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "catboost" / "wine_reranker.cbm"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIEWS = ("original", "bottle_crop", "label_crop")
VIEW_COLLECTIONS = {
    "original": "wine_original_siglip2",
    "bottle_crop": "wine_bottle_crop_siglip2",
    "label_crop": "wine_label_crop_siglip2",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalImage:
    expected: str
    path: Path


def collect_eval_images(eval_dir: Path, limit: int | None = None) -> list[EvalImage]:
    items: list[EvalImage] = []
    for folder in sorted(eval_dir.iterdir()):
        if not folder.is_dir():
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                items.append(EvalImage(expected=folder.name, path=image_path))
    return items[:limit] if limit is not None else items


def normalize_weights(weights: dict[str, float], active_views: list[str]) -> dict[str, float]:
    total = sum(weights.get(view, 0.0) for view in active_views)
    if total <= 0:
        return {}
    return {view: weights.get(view, 0.0) / total for view in active_views}


def point_payload(point: Any) -> dict[str, Any]:
    return getattr(point, "payload", None) or {}


def group_qdrant_points(points: list[Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    slugs: dict[str, str] = {}
    for rank, point in enumerate(points, start=1):
        payload = point_payload(point)
        wine_id = str(payload["wine_id"]) if payload.get("wine_id") else None
        if wine_id is None:
            continue
        point_id = str(getattr(point, "id", ""))
        photo_id = str(payload.get("photo_id") or point_id)
        score = float(getattr(point, "score", 0.0) or 0.0)
        grouped.setdefault(wine_id, []).append(
            {
                "score": score,
                "rank": rank,
                "point_id": point_id,
                "photo_id": photo_id,
            }
        )
        if payload.get("slug"):
            slugs.setdefault(wine_id, str(payload["slug"]))
    return grouped, slugs


def crop_status(query_views, active_views: list[str]) -> str:
    parts: list[str] = []
    for view in VIEWS:
        crop = query_views.crops.get(view)
        if crop is None:
            continue
        available = bool(getattr(crop, "available", False))
        confidence = getattr(crop, "confidence", None)
        box = getattr(crop, "box", None)
        marker = "active" if view in active_views else "inactive"
        if confidence is None:
            parts.append(f"{view}={available}/{marker}")
        else:
            parts.append(f"{view}={available}/{marker} conf={confidence:.3f} box={box}")
    return "; ".join(parts)


async def extract_ocr_text(llm: ChatOpenAIWrapper, query_views) -> tuple[str, str]:
    source_view = WineService._ocr_source_view(query_views)
    image = query_views.images.get(source_view)
    if image is None:
        return "", source_view
    text = await llm.ocr_image_text(WineService._image_to_jpeg_bytes(image))
    return text, source_view


async def build_ocr_feature_map(
    *,
    repository: WineRepository,
    ocr_text: str,
    wine_ids: list[str],
) -> dict[str, dict[str, float]]:
    if not ocr_text or not wine_ids:
        return {}

    from src.ml.text_normalization import normalize_match_text

    normalized_ocr = normalize_match_text(ocr_text)
    if not normalized_ocr:
        return {}

    wines = await repository.load_wines_by_ids(wine_ids)
    raw_color_aliases = await repository.load_color_aliases()
    color_aliases_by_color = {
        normalize_match_text(color): WineService._dedupe_text_variants([color, *aliases])
        for color, aliases in raw_color_aliases.items()
        if normalize_match_text(color)
    }
    wine_by_id = {str(wine.id): wine for wine in wines}
    result: dict[str, dict[str, float]] = {}
    for wine_id in wine_ids:
        wine = wine_by_id.get(wine_id)
        candidate_text = WineService._wine_to_ocr_candidate_text(wine)
        normalized_candidate = normalize_match_text(candidate_text)
        wratio = fuzz.WRatio(normalized_ocr, normalized_candidate) if normalized_candidate else 0.0
        token_set = fuzz.token_set_ratio(normalized_ocr, normalized_candidate) if normalized_candidate else 0.0
        fuzzy_score = (
            OCR_FUZZY_WRATIO_WEIGHT * wratio
            + OCR_FUZZY_TOKEN_SET_WEIGHT * token_set
        ) / 100.0
        structured_score = WineService._structured_ocr_score(normalized_ocr, wine, color_aliases_by_color)
        ocr_score = OCR_DOMAIN_WEIGHT * structured_score["domain_score"] + OCR_FUZZY_WEIGHT * fuzzy_score
        result[wine_id] = {
            "ocr_applied": 1.0,
            "ocr_score": float(ocr_score),
            "ocr_domain_score": float(structured_score["domain_score"]),
            "ocr_fuzzy_score": float(fuzzy_score),
            "ocr_grape_score": float(structured_score["grape_score"]),
            "ocr_grape_match_count": float(structured_score["grape_match_count"]),
            "ocr_grape_total_count": float(structured_score["grape_total_count"]),
            "ocr_grape_missing_count": float(structured_score["grape_missing_count"]),
            "ocr_grape_coverage": float(structured_score["grape_coverage"]),
            "ocr_producer_score": float(structured_score["producer_score"]),
            "ocr_name_score": float(structured_score["name_score"]),
            "ocr_color_score": float(structured_score["color_score"]),
            "ocr_color_detected_count": float(structured_score["color_detected_count"]),
            "ocr_color_detected_match": float(structured_score["color_detected_match"]),
            "ocr_color_detected_mismatch": float(structured_score["color_detected_mismatch"]),
            "ocr_sugar_score": float(structured_score["sugar_score"]),
            "ocr_sugar_detected_count": float(structured_score["sugar_detected_count"]),
            "ocr_sugar_detected_match": float(structured_score["sugar_detected_match"]),
            "ocr_sugar_detected_mismatch": float(structured_score["sugar_detected_mismatch"]),
            "ocr_wratio": float(wratio),
            "ocr_token_set_ratio": float(token_set),
            "ocr_wratio_score": float(wratio / 100.0),
            "ocr_token_set_score": float(token_set / 100.0),
            "ocr_text_length": float(len(normalized_ocr)),
            "ocr_candidate_text_length": float(len(normalized_candidate)),
        }
    return result


async def collect_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    images = collect_eval_images(args.eval_dir, args.limit)
    if not images:
        raise RuntimeError(f"No eval images found under {args.eval_dir}")

    label_cropper = YoloLabelCropper(YoloSettings(model_path=str(args.label_model)))
    bottle_cropper = YoloBottleCropper(YoloSettings(model_path=str(args.bottle_model)))
    photo_service = WinePhotoService(label_cropper=label_cropper, bottle_cropper=bottle_cropper)
    embedder = SiglipImageEmbedder(
        all_settings.embeddings.model_copy(
            update={
                "model_dir": str(args.siglip_model_dir),
                "device": args.device,
            }
        )
    )
    qdrant = QdrantClient(all_settings.qdrant)
    database = DatabaseClient(all_settings.database) if args.with_ocr else None
    repository = WineRepository(database) if database is not None else None
    llm = ChatOpenAIWrapper(all_settings.llm) if args.with_ocr else None

    await label_cropper.start()
    if bottle_cropper is not None:
        await bottle_cropper.start()
    await embedder.start()
    await qdrant.connect()
    if database is not None:
        await database.connect()
    if llm is not None:
        await llm.start()

    records: list[dict[str, Any]] = []
    stats = {
        "label_crops": 0,
        "bottle_crops": 0,
        "embedding_seconds": 0.0,
        "qdrant_seconds": 0.0,
        "ocr_seconds": 0.0,
        "ocr_queries": 0,
        "ocr_candidates": 0,
    }
    started = perf_counter()
    print(
        "collecting CatBoost training features: "
        f"eval_images={len(images)} topk={args.topk} "
        f"candidate_pool={args.candidate_pool} "
        f"with_bottle_crop=True "
        f"with_patches=False with_ocr={args.with_ocr}"
    )
    try:
        for idx, item in enumerate(images, start=1):
            item_started = perf_counter()
            image_bytes = item.path.read_bytes()
            query_views = photo_service.build_query_views(image_bytes)
            active_views = [
                view
                for view in VIEWS
                if view in query_views.images
                and query_views.crops.get(view) is not None
                and query_views.crops[view].available
            ]
            label_area_ratio = WineService._label_crop_area_ratio(query_views.crops)
            label_confidence = WineService._label_crop_confidence(query_views.crops)
            label_photo_mode = (
                label_area_ratio > LABEL_PHOTO_AREA_THRESHOLD
                and label_confidence > LABEL_PHOTO_CONFIDENCE_THRESHOLD
            )
            if label_photo_mode and "label_crop" in active_views:
                active_views = [view for view in active_views if view == "label_crop"]
            if query_views.crops.get("label_crop") is not None and query_views.crops["label_crop"].available:
                stats["label_crops"] += 1
            if query_views.crops.get("bottle_crop") is not None and query_views.crops["bottle_crop"].available:
                stats["bottle_crops"] += 1
            if not active_views:
                logger.warning("No active views for %s", item.path)
                continue

            embed_started = perf_counter()
            vectors = embedder.embed_many([query_views.images[view] for view in active_views])
            embed_elapsed = perf_counter() - embed_started
            stats["embedding_seconds"] += embed_elapsed
            qdrant_started = perf_counter()
            responses = await asyncio.gather(
                *[
                    qdrant.search_collection(
                        collection_name=VIEW_COLLECTIONS[view],
                        vector=vector,
                        limit=args.topk,
                        with_payload=True,
                    )
                    for view, vector in zip(active_views, vectors, strict=True)
                ]
            )
            qdrant_elapsed = perf_counter() - qdrant_started
            stats["qdrant_seconds"] += qdrant_elapsed

            wine_view_matches: dict[str, dict[str, list[dict[str, Any]]]] = {}
            wine_slugs: dict[str, str] = {}
            per_view_counts: dict[str, int] = {}
            for view, response in zip(active_views, responses, strict=True):
                points = list(getattr(response, "points", []) or []) if response is not None else []
                per_view_counts[view] = len(points)
                grouped, slugs = group_qdrant_points(points)
                for wine_id, matches in grouped.items():
                    wine_view_matches.setdefault(wine_id, {}).setdefault(view, []).extend(matches)
                wine_slugs.update(slugs)

            weights = normalize_weights(
                VIEW_WEIGHTS,
                [view for view in active_views if per_view_counts.get(view, 0) > 0],
            )
            base_rows = build_candidate_feature_rows(
                wine_view_matches=wine_view_matches,
                wine_slugs=wine_slugs,
                weights=weights,
                crops=query_views.crops,
            )
            base_rows.sort(key=lambda row: row.baseline_score, reverse=True)
            feature_candidate_ids = {row.wine_id for row in base_rows[: args.candidate_pool]}

            ocr_features = {}
            ocr_elapsed = 0.0
            ocr_source_view = None
            if llm is not None and repository is not None:
                ocr_started = perf_counter()
                try:
                    ocr_text, ocr_source_view = await extract_ocr_text(llm, query_views)
                    ocr_features = await build_ocr_feature_map(
                        repository=repository,
                        ocr_text=ocr_text,
                        wine_ids=list(feature_candidate_ids),
                    )
                except Exception:
                    logger.exception("OCR feature extraction failed for %s", item.path)
                ocr_elapsed = perf_counter() - ocr_started
                stats["ocr_seconds"] += ocr_elapsed
                stats["ocr_queries"] += 1
                stats["ocr_candidates"] += len(ocr_features)

            rows = build_candidate_feature_rows(
                wine_view_matches=wine_view_matches,
                wine_slugs=wine_slugs,
                weights=weights,
                crops=query_views.crops,
                ocr_features=ocr_features,
            )
            rows = [row for row in rows if row.wine_id in feature_candidate_ids]
            for row in rows:
                records.append(
                    {
                        "query_id": f"{item.expected}/{item.path.name}",
                        "expected": item.expected,
                        "image": str(item.path.relative_to(PROJECT_ROOT)),
                        "wine_id": row.wine_id,
                        "slug": row.slug,
                        "label": 1 if row.wine_id == item.expected else 0,
                        "baseline_score": row.baseline_score,
                        "features": row.features,
                    }
                )

            if args.log_each_image:
                print(
                    f"[{idx}/{len(images)}] {item.path.relative_to(PROJECT_ROOT)} "
                    f"active_views={active_views} qdrant_counts={per_view_counts} "
                    f"label_photo_mode={label_photo_mode} candidates={len(rows)} "
                    f"embed={embed_elapsed:.2f}s "
                    f"qdrant={qdrant_elapsed:.2f}s "
                    f"ocr={ocr_elapsed:.2f}s ocr_source={ocr_source_view or '-'} "
                    f"ocr_candidates={len(ocr_features)} "
                    f"total={perf_counter() - item_started:.2f}s"
                )
                print(f"  crops: {crop_status(query_views, active_views)}")
            if idx % args.progress_every == 0 or idx == len(images):
                print(
                    f"collected {idx}/{len(images)} images, rows={len(records)}, "
                    f"label_crops={stats['label_crops']}, bottle_crops={stats['bottle_crops']}, "
                    f"ocr_candidates={stats['ocr_candidates']}, "
                    f"embed={stats['embedding_seconds']:.1f}s, "
                    f"qdrant={stats['qdrant_seconds']:.1f}s, "
                    f"ocr={stats['ocr_seconds']:.1f}s, "
                    f"elapsed={perf_counter() - started:.1f}s"
                )
    finally:
        if llm is not None:
            await llm.stop()
        if database is not None:
            await database.close()
        await qdrant.close()
        await embedder.stop()
        await bottle_cropper.stop()
        await label_cropper.stop()

    print(
        "feature collection finished: "
        f"images={len(images)} rows={len(records)} "
        f"label_crops={stats['label_crops']} bottle_crops={stats['bottle_crops']} "
        f"ocr_queries={stats['ocr_queries']} "
        f"ocr_candidates={stats['ocr_candidates']} "
        f"embed_seconds={stats['embedding_seconds']:.1f} "
        f"qdrant_seconds={stats['qdrant_seconds']:.1f} "
        f"ocr_seconds={stats['ocr_seconds']:.1f}"
    )
    return records


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_dataset(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def split_query_ids(records: list[dict[str, Any]], valid_fraction: float) -> tuple[set[str], set[str]]:
    query_ids = sorted({record["query_id"] for record in records})
    if len(query_ids) < 2:
        return set(query_ids), set()
    rng = np.random.default_rng(42)
    shuffled = list(query_ids)
    rng.shuffle(shuffled)
    valid_size = max(1, int(round(len(shuffled) * valid_fraction)))
    valid = set(shuffled[:valid_size])
    train = set(shuffled[valid_size:])
    if not train:
        train, valid = valid, set()
    return train, valid


def rank_metrics(records: list[dict[str, Any]], scores: list[float]) -> dict[str, float]:
    by_query: dict[str, list[tuple[str, int, float]]] = {}
    for record, score in zip(records, scores, strict=True):
        by_query.setdefault(record["query_id"], []).append((record["wine_id"], int(record["label"]), float(score)))
    ranks: list[int | None] = []
    for rows in by_query.values():
        rows.sort(key=lambda item: item[2], reverse=True)
        rank = next((idx for idx, (_wine_id, label, _score) in enumerate(rows, start=1) if label == 1), None)
        ranks.append(rank)
    n = len(ranks)
    return {
        "queries": float(n),
        "acc1": sum(1 for rank in ranks if rank == 1) / n if n else 0.0,
        "mrr": sum(1.0 / rank for rank in ranks if rank) / n if n else 0.0,
        "recall5": sum(1 for rank in ranks if rank is not None and rank <= 5) / n if n else 0.0,
        "missing": float(sum(1 for rank in ranks if rank is None)),
    }


def build_query_report(
    records: list[dict[str, Any]],
    baseline_scores: list[float],
    model_scores: list[float],
) -> list[dict[str, Any]]:
    by_query: dict[str, list[tuple[dict[str, Any], float, float]]] = {}
    for record, baseline_score, model_score in zip(records, baseline_scores, model_scores, strict=True):
        by_query.setdefault(record["query_id"], []).append((record, float(baseline_score), float(model_score)))

    rows: list[dict[str, Any]] = []
    for query_id, items in sorted(by_query.items()):
        expected = str(items[0][0]["expected"])
        image = str(items[0][0]["image"])
        baseline_ranked = sorted(items, key=lambda item: item[1], reverse=True)
        model_ranked = sorted(items, key=lambda item: item[2], reverse=True)
        baseline_rank = next(
            (idx for idx, (record, _baseline, _model) in enumerate(baseline_ranked, start=1) if int(record["label"]) == 1),
            None,
        )
        model_rank = next(
            (idx for idx, (record, _baseline, _model) in enumerate(model_ranked, start=1) if int(record["label"]) == 1),
            None,
        )
        baseline_top = baseline_ranked[0]
        model_top = model_ranked[0]
        baseline_correct = baseline_rank == 1
        model_correct = model_rank == 1
        if model_correct and not baseline_correct:
            verdict = "fixed"
        elif baseline_correct and not model_correct:
            verdict = "lost"
        elif model_rank is not None and (baseline_rank is None or model_rank < baseline_rank):
            verdict = "improved"
        elif baseline_rank is not None and (model_rank is None or model_rank > baseline_rank):
            verdict = "worsened"
        else:
            verdict = "same"
        rows.append(
            {
                "query_id": query_id,
                "image": image,
                "expected": expected,
                "verdict": verdict,
                "baseline_rank": baseline_rank,
                "catboost_rank": model_rank,
                "baseline_top1": baseline_top[0]["wine_id"],
                "catboost_top1": model_top[0]["wine_id"],
                "baseline_top1_score": baseline_top[1],
                "catboost_top1_score": model_top[2],
                "baseline_expected_score": next(
                    (score for record, score, _model_score in items if int(record["label"]) == 1),
                    None,
                ),
                "catboost_expected_score": next(
                    (score for record, _baseline_score, score in items if int(record["label"]) == 1),
                    None,
                ),
                "candidates": len(items),
            }
        )
    return rows


def print_query_report_summary(rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["verdict"])] = counts.get(str(row["verdict"]), 0) + 1
    print(
        "delta        "
        f"fixed={counts.get('fixed', 0)} "
        f"lost={counts.get('lost', 0)} "
        f"improved={counts.get('improved', 0)} "
        f"worsened={counts.get('worsened', 0)} "
        f"same={counts.get('same', 0)}"
    )


def save_query_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "image",
        "expected",
        "verdict",
        "baseline_rank",
        "catboost_rank",
        "baseline_top1",
        "catboost_top1",
        "baseline_top1_score",
        "catboost_top1_score",
        "baseline_expected_score",
        "catboost_expected_score",
        "candidates",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_feature_importance(path: Path, feature_names: list[str], importances: list[float], top_n: int) -> None:
    rows = sorted(
        (
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(feature_names, importances, strict=True)
        ),
        key=lambda row: row["importance"],
        reverse=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["feature", "importance"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"top {min(top_n, len(rows))} feature importance:")
    for row in rows[:top_n]:
        print(f"  {row['feature']:<34} {row['importance']:.6f}")


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name:<12} queries={metrics['queries']:.0f} "
        f"Acc@1={metrics['acc1']:.2%} MRR={metrics['mrr']:.4f} "
        f"R@5={metrics['recall5']:.2%} missing={metrics['missing']:.0f}"
    )


def train_model(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    try:
        from catboost import CatBoostRanker, Pool
    except ImportError as exc:
        raise RuntimeError("catboost is not installed. Run `uv add catboost` or `pip install catboost`.") from exc

    feature_names = sorted({name for record in records for name in record["features"]})
    train_queries, valid_queries = split_query_ids(records, args.valid_fraction)
    train = sorted(
        (record for record in records if record["query_id"] in train_queries),
        key=lambda record: (record["query_id"], record["wine_id"]),
    )
    valid = sorted(
        (record for record in records if record["query_id"] in valid_queries),
        key=lambda record: (record["query_id"], record["wine_id"]),
    )

    def matrix(rows: list[dict[str, Any]]) -> list[list[float]]:
        return [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows]

    train_pool = Pool(
        data=matrix(train),
        label=[int(row["label"]) for row in train],
        group_id=[row["query_id"] for row in train],
        feature_names=feature_names,
    )
    eval_set = None
    if valid:
        eval_set = Pool(
            data=matrix(valid),
            label=[int(row["label"]) for row in valid],
            group_id=[row["query_id"] for row in valid],
            feature_names=feature_names,
        )

    model = CatBoostRanker(
        loss_function="YetiRank",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        random_seed=42,
        verbose=args.verbose_eval,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=eval_set)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.model_output))
    metadata = {
        "feature_names": feature_names,
        "loss_function": "YetiRank",
        "train_queries": len(train_queries),
        "valid_queries": len(valid_queries),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "feature_count": len(feature_names),
        "per_view_top_k": args.topk,
        "candidate_pool": args.candidate_pool,
        "with_patches": False,
        "with_bottle_crop": True,
        "with_ocr": args.with_ocr,
    }
    args.model_output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved model: {args.model_output}")
    print(f"saved metadata: {args.model_output.with_suffix('.json')}")
    report_records = valid or train
    report_matrix = matrix(report_records)
    baseline_scores = [float(row["baseline_score"]) for row in report_records]
    model_scores = [float(score) for score in model.predict(report_matrix)]
    print_metrics("baseline", rank_metrics(report_records, baseline_scores))
    print_metrics("catboost", rank_metrics(report_records, model_scores))

    query_report = build_query_report(report_records, baseline_scores, model_scores)
    print_query_report_summary(query_report)
    report_path = args.model_output.with_name(f"{args.model_output.stem}_validation_report.csv")
    save_query_report(report_path, query_report)
    print(f"saved validation report: {report_path}")

    importance_path = args.model_output.with_name(f"{args.model_output.stem}_feature_importance.csv")
    importances = model.get_feature_importance(eval_set or train_pool).tolist()
    save_feature_importance(importance_path, feature_names, importances, args.top_features)
    print(f"saved feature importance: {importance_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Qdrant features from eval images and train CatBoost reranker.")
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--siglip-model-dir", type=Path, default=PROJECT_ROOT / "models" / "siglip2")
    parser.add_argument("--label-model", type=Path, default=PROJECT_ROOT / "models" / "yolo" / "label.pt")
    parser.add_argument("--bottle-model", type=Path, default=PROJECT_ROOT / "models" / "yolo" / "yolo26x.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=PER_VIEW_TOP_K)
    parser.add_argument("--candidate-pool", type=int, default=GLOBAL_CANDIDATE_LIMIT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--log-each-image", action="store_true")
    parser.add_argument("--with-bottle-crop", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--with-patches", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dinov3-model-dir", type=Path, default=PROJECT_ROOT / "models" / "dinov3", help=argparse.SUPPRESS)
    parser.add_argument("--with-ocr", action="store_true")
    parser.add_argument("--use-existing-dataset", action="store_true")
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--verbose-eval", type=int, default=100)
    parser.add_argument("--top-features", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.use_existing_dataset:
        records = load_dataset(args.dataset_output)
    else:
        records = asyncio.run(collect_dataset(args))
        save_dataset(args.dataset_output, records)
        print(f"saved dataset: {args.dataset_output}")
    train_model(records, args)


if __name__ == "__main__":
    main()
