#!/usr/bin/env python3
"""
Grid search по весам cosine-агрегации на eval-наборе.

Для каждого eval-изображения строятся такие же query views, как в API
(original, bottle_crop, label_crop), затем напрямую запрашивается Qdrant и
сохраняются top-k кандидаты по каждому view.
Затем локально перебираются комбинации весов, лимитов и параметров агрегации,
после чего считаются Acc@1 / MRR / Recall@10 после такого же wine-level fusion,
как в WineService.

Использование:
  python scripts/grid_search_weights.py [--limit N]
  python scripts/grid_search_weights.py --reuse-ranks
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services.photo_service import WinePhotoService
from src.api.services.wine_service import (
    GLOBAL_CANDIDATE_LIMIT,
    LABEL_PHOTO_AREA_THRESHOLD,
    LABEL_PHOTO_CONFIDENCE_THRESHOLD,
    PER_VIEW_TOP_K,
    VIEW_SCORE_BEST_WEIGHT,
    VIEW_SCORE_MEAN_WEIGHT,
    VIEW_SCORE_TOP_K_PHOTOS,
    VIEW_WEIGHTS,
    WineService,
)
from src.connections.qdrant import QdrantClient
from src.ml.dinov3 import DinoV3ImageEmbedder
from src.ml.siglip2 import SiglipImageEmbedder
from src.ml.yolo import YoloBottleCropper, YoloLabelCropper
from src.settings.settings import YoloSettings, all_settings

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

VIEWS = ("original", "bottle_crop", "label_crop")
VIEW_INDEX = {view: idx for idx, view in enumerate(VIEWS)}

# Текущие runtime-веса WineService. Они явно входят в сетку ниже.
DEFAULT_VIEW_WEIGHTS = VIEW_WEIGHTS.copy()

# Сетка VIEW_WEIGHTS. Значения потом нормализуются по активным views,
# как в WineService._normalized_weights.
VIEW_WEIGHT_VALUES = {
    "original": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
    "bottle_crop": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "label_crop": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
}

# Сетка runtime-параметров WineService.
GLOBAL_CANDIDATE_LIMITS = [25, GLOBAL_CANDIDATE_LIMIT, 75, 100]
PER_VIEW_TOP_KS = [50, 100, PER_VIEW_TOP_K]

# score = agg_max_weight * max(view_scores) + agg_sum_weight * sum(view_scores)
AGGREGATION_WEIGHTS = [
    (1.0, 0.0),
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
]

def normalized_weight_array(weights: dict[str, float], active_mask: np.ndarray) -> np.ndarray:
    values = np.array([weights.get(view, 0.0) for view in VIEWS], dtype=np.float64)
    values = np.where(active_mask, values, 0.0)
    total = values.sum()
    if total <= 0:
        return values
    return values / total


def normalized_weight_dict(weights: dict[str, float]) -> dict[str, float]:
    active_mask = np.array([weights.get(view, 0.0) > 0 for view in VIEWS], dtype=bool)
    normalized = normalized_weight_array(weights, active_mask)
    return {view: float(normalized[idx]) for idx, view in enumerate(VIEWS)}


def iter_view_weight_grid() -> list[dict[str, float]]:
    seen: set[tuple[float, float, float]] = set()
    grid: list[dict[str, float]] = []
    for original, bottle, label in itertools.product(
        VIEW_WEIGHT_VALUES["original"],
        VIEW_WEIGHT_VALUES["bottle_crop"],
        VIEW_WEIGHT_VALUES["label_crop"],
    ):
        if original + bottle + label <= 0:
            continue
        key = (original, bottle, label)
        if key in seen:
            continue
        seen.add(key)
        grid.append(
            {
                "original": original,
                "bottle_crop": bottle,
                "label_crop": label,
            }
        )
    default_key = (
        DEFAULT_VIEW_WEIGHTS["original"],
        DEFAULT_VIEW_WEIGHTS["bottle_crop"],
        DEFAULT_VIEW_WEIGHTS["label_crop"],
    )
    if default_key not in seen:
        grid.append(DEFAULT_VIEW_WEIGHTS.copy())
    return grid


def prepare_records(records: list[dict], per_view_top_ks: list[int]) -> dict[int, list[dict]]:
    prepared: dict[int, list[dict]] = {topk: [] for topk in per_view_top_ks}
    for topk in per_view_top_ks:
        for record in records:
            wine_to_scores: dict[str, list[float]] = {}
            view_has_candidates = np.zeros(len(VIEWS), dtype=bool)
            for view in VIEWS:
                view_idx = VIEW_INDEX[view]
                candidates = record["views"].get(view, [])[:topk]
                if candidates:
                    view_has_candidates[view_idx] = True
                for candidate in candidates:
                    wine_id = str(candidate["wine_id"])
                    score = float(candidate.get("score") or 0.0)
                    scores = wine_to_scores.setdefault(wine_id, [0.0] * len(VIEWS))
                    # В runtime на wine/view остаётся один лучший view-score.
                    scores[view_idx] = max(scores[view_idx], score)

            wine_ids = list(wine_to_scores)
            expected = str(record["expected"])
            expected_index = wine_ids.index(expected) if expected in wine_to_scores else None
            score_matrix = np.array(
                [wine_to_scores[wine_id] for wine_id in wine_ids],
                dtype=np.float64,
            )
            if score_matrix.size == 0:
                score_matrix = np.zeros((0, len(VIEWS)), dtype=np.float64)
            prepared[topk].append(
                {
                    "expected_index": expected_index,
                    "score_matrix": score_matrix,
                    "view_has_candidates": view_has_candidates,
                }
            )
    return prepared


def collect_images(eval_dir: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    if not eval_dir.is_dir():
        print(f"Ошибка: директория {eval_dir} не найдена", file=sys.stderr)
        return items
    for folder in sorted(eval_dir.iterdir()):
        if not folder.is_dir():
            continue
        wine_id = folder.name
        for img in sorted(folder.iterdir()):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                items.append((wine_id, img))
    return items


def view_collection(view: str) -> str:
    return f"wine_{view}_{all_settings.search.collection_encoder}"


def group_view_points(points: list[Any]) -> list[dict]:
    by_wine: dict[str, dict[str, Any]] = {}
    for rank, point in enumerate(points, start=1):
        payload = getattr(point, "payload", None) or {}
        wine_id = str(payload["wine_id"]) if payload.get("wine_id") else None
        if wine_id is None:
            continue

        cosine = float(getattr(point, "score", 0.0) or 0.0)
        photo_id = str(payload.get("photo_id") or getattr(point, "id", ""))
        current = by_wine.get(wine_id)
        if current is None:
            by_wine[wine_id] = {
                "wine_id": wine_id,
                "slug": str(payload.get("slug") or wine_id),
                "rank": rank,
                "photo_id": photo_id,
                "photo_scores": [cosine],
            }
            continue

        current["photo_scores"].append(cosine)
        if rank < int(current["rank"]):
            current["rank"] = rank
            current["photo_id"] = photo_id

    results: list[dict] = []
    for row in by_wine.values():
        scores = sorted(row["photo_scores"], reverse=True)
        top_scores = scores[:VIEW_SCORE_TOP_K_PHOTOS]
        view_score = (
            VIEW_SCORE_BEST_WEIGHT * scores[0]
            + VIEW_SCORE_MEAN_WEIGHT * mean(top_scores)
        )
        results.append(
            {
                "wine_id": row["wine_id"],
                "slug": row["slug"],
                "rank": row["rank"],
                "score": view_score,
                "cosine_score": scores[0],
                "photo_id": row["photo_id"],
                "n_photos": len(scores),
            }
        )

    return sorted(results, key=lambda item: float(item["score"]), reverse=True)


def close_query_images(query_views) -> None:
    seen: set[int] = set()
    for image in query_views.images.values():
        image_id = id(image)
        if image_id in seen:
            continue
        seen.add(image_id)
        image.close()


def build_embedder() -> SiglipImageEmbedder | DinoV3ImageEmbedder:
    if all_settings.search.global_encoder == "dinov3":
        return DinoV3ImageEmbedder(all_settings.dinov3)
    return SiglipImageEmbedder(all_settings.embeddings)


async def collect_candidates(
    images: list[tuple[str, Path]],
    topk: int,
) -> list[dict]:
    """Для каждого изображения собирает top-k кандидатов по каждому view."""
    records: list[dict] = []

    label_cropper = YoloLabelCropper(
        YoloSettings(model_path='models/yolo/label.pt')
    )
    bottle_cropper = YoloBottleCropper(
        YoloSettings(model_path='models/yolo/yolo26x.pt')
    )
    photo_service = WinePhotoService(
        label_cropper=label_cropper,
        bottle_cropper=bottle_cropper,
    )
    embedder = build_embedder()
    qdrant = QdrantClient(all_settings.qdrant)

    try:
        await label_cropper.start()
        await bottle_cropper.start()
        await embedder.start()
        await qdrant.connect()

        for idx, (expected_id, img_path) in enumerate(images, start=1):
            record: dict = {
                "expected": expected_id,
                "image": img_path.name,
                "path": str(img_path.relative_to(PROJECT_ROOT)),
                "views": {view: [] for view in VIEWS},
                "active_views": [],
                "label_photo_mode": False,
            }
            query_views = None
            try:
                query_views = photo_service.build_query_views(img_path.read_bytes())
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

                vectors = embedder.embed_many([query_views.images[view] for view in active_views])
                search_tasks = [
                    qdrant.search_collection(
                        collection_name=view_collection(view),
                        vector=vector,
                        limit=topk,
                        with_payload=True,
                    )
                    for view, vector in zip(active_views, vectors, strict=True)
                ]
                responses = await asyncio.gather(*search_tasks)
                for view, response in zip(active_views, responses, strict=True):
                    points = list(getattr(response, "points", []) or []) if response is not None else []
                    record["views"][view] = group_view_points(points)

                record["active_views"] = active_views
                record["label_photo_mode"] = label_photo_mode
                record["label_area_ratio"] = label_area_ratio
                record["label_confidence"] = label_confidence
            except Exception as exc:  # noqa: BLE001
                print(f"[{idx}/{len(images)}] ОШИБКА {img_path.name}: {exc}")
            finally:
                if query_views is not None:
                    close_query_images(query_views)
            records.append(record)
            if idx % 50 == 0 or idx == len(images):
                print(f"Собрано кандидатов: {idx}/{len(images)}")
    finally:
        await qdrant.close()
        await embedder.stop()
        await bottle_cropper.stop()
        await label_cropper.stop()
    return records


def evaluate_ranks(
    prepared_records: list[dict],
    weights: dict[str, float],
    agg_max_weight: float,
    agg_sum_weight: float,
) -> list[int | None]:
    ranks: list[int | None] = []
    positive_weight_mask = np.array([weights.get(view, 0.0) > 0 for view in VIEWS], dtype=bool)
    for record in prepared_records:
        expected_index = record["expected_index"]
        scores = record["score_matrix"]
        if expected_index is None or scores.shape[0] == 0:
            ranks.append(None)
            continue

        active_mask = record["view_has_candidates"] & positive_weight_mask
        if not active_mask.any():
            ranks.append(None)
            continue

        normalized = normalized_weight_array(weights, active_mask)
        weighted_scores = scores[:, active_mask] * normalized[active_mask]
        max_scores = weighted_scores.max(axis=1)
        sum_scores = weighted_scores.sum(axis=1)
        fused_scores = agg_max_weight * max_scores + agg_sum_weight * sum_scores

        expected_score = fused_scores[expected_index]
        higher = np.count_nonzero(fused_scores > expected_score)
        earlier_ties = np.count_nonzero(
            (fused_scores[:expected_index] == expected_score)
        )
        ranks.append(int(higher + earlier_ties + 1))
    return ranks


def metrics_from_ranks(ranks: list[int | None], global_candidate_limit: int) -> dict:
    n = len(ranks)
    if n == 0:
        return {"acc1": 0.0, "mrr": 0.0, "recall10": 0.0}

    acc1 = 0
    mrr_sum = 0.0
    recall10 = 0
    recall_limit = min(10, global_candidate_limit)
    for rank in ranks:
        if rank is None or rank > global_candidate_limit:
            continue
        if rank == 1:
            acc1 += 1
        mrr_sum += 1.0 / rank
        if rank <= recall_limit:
            recall10 += 1
    return {
        "acc1": acc1 / n,
        "mrr": mrr_sum / n,
        "recall10": recall10 / n,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Grid search по весам cosine-агрегации")
    parser.add_argument(
        "--api-url",
        default=None,
        help="Deprecated: кандидаты теперь собираются напрямую из Qdrant.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--topk", type=int, default=max(PER_VIEW_TOP_KS))
    parser.add_argument("--reuse-ranks", action="store_true")
    parser.add_argument(
        "--ranks-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "grid_search_ranks.json",
    )
    args = parser.parse_args()
    if args.api_url is not None:
        print("--api-url игнорируется: кандидаты собираются напрямую из Qdrant.")

    if args.reuse_ranks:
        if not args.ranks_path.is_file():
            print(f"Ошибка: файл кандидатов {args.ranks_path} не найден", file=sys.stderr)
            sys.exit(1)
        records = json.loads(args.ranks_path.read_text(encoding="utf-8"))
        if args.limit is not None:
            records = records[: args.limit]
        print(f"Кандидаты загружены из {args.ranks_path}: {len(records)} записей")
    else:
        images = collect_images(EVAL_DIR)
        if not images:
            print("Нет изображений для оценки.", file=sys.stderr)
            sys.exit(1)
        if args.limit is not None:
            images = images[: args.limit]

        print(f"Всего изображений: {len(images)}")
        print(
            "Runtime config: "
            f"global_encoder={all_settings.search.global_encoder}, "
            f"collection_encoder={all_settings.search.collection_encoder}, "
            f"collections={[view_collection(view) for view in VIEWS]}"
        )
        print("Оценивается только visual global fusion, без OCR/patch/LLM rerank.")
        collect_topk = max(args.topk, max(PER_VIEW_TOP_KS))
        print(f"Собираю кандидатов по views: {VIEWS} (topk={collect_topk})...")
        records = asyncio.run(collect_candidates(images, collect_topk))

        # Сохраняем кандидатов, чтобы не перезапрашивать Qdrant.
        args.ranks_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Кандидаты сохранены в {args.ranks_path}")

    prepared_by_topk = prepare_records(records, PER_VIEW_TOP_KS)

    # Grid search.
    view_weight_grid = iter_view_weight_grid()
    rank_combinations = (
        len(view_weight_grid)
        * len(PER_VIEW_TOP_KS)
        * len(AGGREGATION_WEIGHTS)
    )
    total_combinations = rank_combinations * len(GLOBAL_CANDIDATE_LIMITS)
    print(
        "Перебираю комбинации: "
        f"VIEW_WEIGHTS={len(view_weight_grid)}, "
        f"PER_VIEW_TOP_K={len(PER_VIEW_TOP_KS)}, "
        f"GLOBAL_CANDIDATE_LIMIT={len(GLOBAL_CANDIDATE_LIMITS)}, "
        f"aggregation={len(AGGREGATION_WEIGHTS)} "
        f"(rank_calculations={rank_combinations}, total_metrics={total_combinations})"
    )
    results: list[dict] = []
    for (
        weights,
        per_view_top_k,
        (agg_max_weight, agg_sum_weight),
    ) in itertools.product(
        view_weight_grid,
        PER_VIEW_TOP_KS,
        AGGREGATION_WEIGHTS,
    ):
        ranks = evaluate_ranks(
            prepared_by_topk[per_view_top_k],
            weights,
            agg_max_weight=agg_max_weight,
            agg_sum_weight=agg_sum_weight,
        )
        for global_candidate_limit in GLOBAL_CANDIDATE_LIMITS:
            metrics = metrics_from_ranks(ranks, global_candidate_limit)
            normalized_weights = normalized_weight_dict(weights)
            results.append(
                {
                    "w_label": weights["label_crop"],
                    "w_bottle": weights["bottle_crop"],
                    "w_original": weights["original"],
                    "nw_label": normalized_weights["label_crop"],
                    "nw_bottle": normalized_weights["bottle_crop"],
                    "nw_original": normalized_weights["original"],
                    "view_weights": weights,
                    "normalized_view_weights": normalized_weights,
                    "per_view_top_k": per_view_top_k,
                    "global_candidate_limit": global_candidate_limit,
                    "agg_max_weight": agg_max_weight,
                    "agg_sum_weight": agg_sum_weight,
                    **metrics,
                }
            )

    # Сортируем по Acc@1, затем по MRR и Recall@10.
    results.sort(key=lambda r: (r["acc1"], r["mrr"], r["recall10"]), reverse=True)
    results_path = Path(__file__).resolve().parent.parent / "data" / "grid_search_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Результаты grid search сохранены в {results_path}")

    print("\n" + "=" * 100)
    print("ТОП-20 КОМБИНАЦИЙ ПО ACC@1")
    print("=" * 100)
    print(
        f"{'w_label':<8} {'w_bottle':<9} {'w_original':<11} "
        f"{'nw_label':<8} {'nw_bottle':<9} {'nw_original':<11} "
        f"{'top_k':<6} {'cand':<6} {'max_w':<6} {'sum_w':<6} "
        f"{'Acc@1':<8} {'MRR':<8} {'Recall@10':<10}"
    )
    print("-" * 100)
    for r in results[:20]:
        print(
            f"{r['w_label']:<8} {r['w_bottle']:<9} {r['w_original']:<11} "
            f"{r['nw_label']:<8.3f} {r['nw_bottle']:<9.3f} {r['nw_original']:<11.3f} "
            f"{r['per_view_top_k']:<6} {r['global_candidate_limit']:<6} "
            f"{r['agg_max_weight']:<6} {r['agg_sum_weight']:<6} "
            f"{r['acc1']:<8.2%} {r['mrr']:<8.4f} {r['recall10']:<10.2%}"
        )

    best = results[0]
    print("\nЛучшие параметры:")
    print(f"  label_crop = {best['w_label']}")
    print(f"  bottle_crop = {best['w_bottle']}")
    print(f"  original = {best['w_original']}")
    print(
        "  normalized = "
        f"label_crop {best['nw_label']:.3f}, "
        f"bottle_crop {best['nw_bottle']:.3f}, "
        f"original {best['nw_original']:.3f}"
    )
    print(f"  PER_VIEW_TOP_K = {best['per_view_top_k']}")
    print(f"  GLOBAL_CANDIDATE_LIMIT = {best['global_candidate_limit']}")
    print(f"  agg_max_weight = {best['agg_max_weight']}")
    print(f"  agg_sum_weight = {best['agg_sum_weight']}")
    print(f"  Acc@1 = {best['acc1']:.2%}, MRR = {best['mrr']:.4f}, Recall@10 = {best['recall10']:.2%}")


if __name__ == "__main__":
    main()
