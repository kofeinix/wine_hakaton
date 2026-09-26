#!/usr/bin/env python3
"""Сбор датасета и обучение CatBoost-реранкера поверх формулы этапа 1.

Запросы (фото с известным вином):
  * eval      — data/eval/<wine_id>/<img>: реальные фото, НЕ в индексе -> валидация;
  * extended  — data/images_extended/<wine_id>/<photo_dir>/original.jpg, по умолчанию yandex_*
                (реальные из веба) и flux_* (сгенерированные). Оба в индексе, поэтому поиск идёт
                leave-one-out: точки этого же фото исключаются фильтром Qdrant, остальные фото
                вина остаются — как у реального пользователя в проде.

  # 1. датасеты (OCR кешируется в data/catboost/ocr_cache.jsonl, сбор можно прервать и продолжить)
  uv run python scripts/train_catboost_reranker.py collect --source eval --with-ocr
  uv run python scripts/train_catboost_reranker.py collect --source extended --with-ocr \
      --max-per-wine 'yandex_*=2' 'flux_*=2'
  # 2. обучение на extended, оценка на реальном eval, финальная модель — на extended + eval
  uv run python scripts/train_catboost_reranker.py train --refit-with-valid \
      --train-dataset data/catboost/extended.jsonl --valid-dataset data/catboost/eval.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import fnmatch
import hashlib
import random
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from qdrant_client import models as qdrant_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services.ocr_match_service import OcrMatchService
from src.api.services.photo_service import WinePhotoService
from src.api.repositories.wine_repository import WineRepository
from src.api.services.visual_search import (
    GLOBAL_CANDIDATE_LIMIT,
    VIEW_WEIGHTS,
    VisualMatches,
    VisualSearcher,
    ViewSelection,
    image_to_jpeg_bytes,
    ocr_source_view,
    select_views,
)
from src.api.services.wine_service import SUGAR_VARIANTS
from src.connections.database.postgres import DatabaseClient
from src.connections.qdrant import QdrantClient
from src.llm.langchain_openai import ChatOpenAIWrapper
from src.ml.catboost_reranker import (
    build_candidate_feature_rows,
    stage1_score,
)
from src.ml.siglip2 import SiglipImageEmbedder
from src.ml.yolo import YoloBottleCropper, YoloLabelCropper
from src.settings.settings import YoloSettings, all_settings


EVAL_DIR = PROJECT_ROOT / "data" / "eval"
EXTENDED_DIR = PROJECT_ROOT / "data" / "images_extended"
DATASET_DIR = PROJECT_ROOT / "data" / "catboost"
DEFAULT_OCR_CACHE = DATASET_DIR / "ocr_cache.jsonl"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "catboost" / "wine_reranker.cbm"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryImage:
    expected: str
    path: Path
    source: str  # eval | vivino | krsk | yandex | flux | ...
    photo_id: str | None  # папка фото в images_extended — для leave-one-out в Qdrant

    @property
    def query_id(self) -> str:
        return str(self.path.relative_to(PROJECT_ROOT))


def photo_source(photo_id: str) -> str:
    return "flux" if photo_id.startswith("flux") else photo_id.split("_")[0]


def collect_eval_queries(eval_dir: Path) -> list[QueryImage]:
    return [
        QueryImage(expected=folder.name, path=image_path, source="eval", photo_id=None)
        for folder in sorted(p for p in eval_dir.iterdir() if p.is_dir())
        for image_path in sorted(folder.iterdir())
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS
    ]


def collect_extended_queries(
    extended_dir: Path,
    patterns: list[str],
    max_per_wine: dict[str, int],
    seed: int,
) -> list[QueryImage]:
    """original.jpg из папок фото по маскам; main никогда не запрос (это якорь индекса)."""
    items: list[QueryImage] = []
    for wine_dir in sorted(p for p in extended_dir.iterdir() if p.is_dir()):
        by_pattern: dict[str, list[Path]] = {}
        for photo_dir in sorted(p for p in wine_dir.iterdir() if p.is_dir() and p.name != "main"):
            pattern = next((pat for pat in patterns if fnmatch.fnmatchcase(photo_dir.name, pat)), None)
            if pattern is not None and (photo_dir / "original.jpg").is_file():
                by_pattern.setdefault(pattern, []).append(photo_dir)
        rng = random.Random(f"{seed}:{wine_dir.name}")  # детерминированная выборка на вино
        for pattern, dirs in by_pattern.items():
            limit = max_per_wine.get(pattern)
            if limit is not None and len(dirs) > limit:
                dirs = sorted(rng.sample(dirs, limit))
            items.extend(
                QueryImage(
                    expected=wine_dir.name,
                    path=photo_dir / "original.jpg",
                    source=photo_source(photo_dir.name),
                    photo_id=photo_dir.name,
                )
                for photo_dir in dirs
            )
    return items


def leave_one_out_filter(item: QueryImage) -> qdrant_models.Filter | None:
    """Исключить из поиска точки самого фото-запроса (все его view)."""
    if item.photo_id is None:
        return None
    return qdrant_models.Filter(
        must_not=[
            qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(key="wine_id", match=qdrant_models.MatchValue(value=item.expected)),
                    qdrant_models.FieldCondition(key="photo_id", match=qdrant_models.MatchValue(value=item.photo_id)),
                ]
            )
        ]
    )


def crop_status(query_views, active_views: list[str]) -> str:
    parts: list[str] = []
    for view in VIEW_WEIGHTS:
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


class OcrCache:
    """OCR-тексты по sha1 JPEG-байтов, отправленных в LLM: повторный сбор не платит за OCR."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.texts: dict[str, str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.texts[row["key"]] = row["text"]
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def get(self, key: str) -> str | None:
        return self.texts.get(key)

    def put(self, key: str, text: str) -> None:
        self.texts[key] = text
        self._file.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


@dataclass
class PreparedQuery:
    item: QueryImage
    matches: VisualMatches
    crops: dict[str, Any]
    candidate_ids: list[str]
    ocr_source_view: str
    ocr_jpeg: bytes | None


def prepare_query(
    item: QueryImage,
    photo_service: WinePhotoService,
    visual: VisualSearcher,
) -> tuple[Any, ViewSelection, list[list[float]]]:
    """CPU/GPU-часть (YOLO + эмбеддинги) — выполняется в потоке, пока идут запросы к LLM."""
    query_views = photo_service.build_query_views(item.path.read_bytes())
    selection = select_views(query_views)
    return query_views, selection, visual.embed(query_views, selection)


async def collect_dataset(args: argparse.Namespace) -> None:
    if args.source == "eval":
        items = collect_eval_queries(args.eval_dir)
    else:
        max_per_wine = dict(
            (pattern, int(limit)) for pattern, limit in (spec.split("=", 1) for spec in args.max_per_wine)
        )
        items = collect_extended_queries(args.extended_dir, args.photo_pattern, max_per_wine, args.seed)
    if args.limit is not None:
        items = items[: args.limit]
    output = args.dataset_output or DATASET_DIR / f"{args.source}.jsonl"
    done = {record["query_id"] for record in load_dataset(output)} if output.exists() else set()
    todo = [item for item in items if item.query_id not in done]
    sources: dict[str, int] = {}
    for item in items:
        sources[item.source] = sources.get(item.source, 0) + 1
    print(f"queries: {len(items)} {sources}; already in {output}: {len(done)}; to collect: {len(todo)}")
    if not todo:
        return

    label_cropper = YoloLabelCropper(YoloSettings(model_path=str(args.label_model)))
    bottle_cropper = YoloBottleCropper(YoloSettings(model_path=str(args.bottle_model)))
    photo_service = WinePhotoService(label_cropper=label_cropper, bottle_cropper=bottle_cropper)
    embedder = SiglipImageEmbedder(
        all_settings.embeddings.model_copy(update={"model_dir": str(args.siglip_model_dir), "device": args.device})
    )
    qdrant = QdrantClient(all_settings.qdrant)
    visual = VisualSearcher(embedder, qdrant, all_settings.search.collection_encoder)
    database = DatabaseClient(all_settings.database) if args.with_ocr else None
    matcher = OcrMatchService(WineRepository(database), SUGAR_VARIANTS) if database is not None else None
    llm = ChatOpenAIWrapper(all_settings.llm) if args.with_ocr else None
    ocr_cache = OcrCache(args.ocr_cache) if args.with_ocr else None
    ocr_semaphore = asyncio.Semaphore(args.ocr_concurrency)

    await label_cropper.start()
    await bottle_cropper.start()
    await embedder.start()
    await qdrant.connect()
    if database is not None:
        await database.connect()
    if llm is not None:
        await llm.start()

    stats = {"queries": 0, "rows": 0, "ocr_calls": 0, "ocr_cached": 0, "ocr_failed": 0, "no_views": 0}
    started = perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    out_file = output.open("a", encoding="utf-8")

    async def ocr_text_for(prepared: PreparedQuery) -> str:
        if llm is None or ocr_cache is None or prepared.ocr_jpeg is None:
            return ""
        key = hashlib.sha1(prepared.ocr_jpeg).hexdigest()
        cached = ocr_cache.get(key)
        if cached is not None:
            stats["ocr_cached"] += 1
            return cached
        async with ocr_semaphore:
            try:
                text = await llm.ocr_image_text(prepared.ocr_jpeg)
            except Exception:
                logger.exception("OCR failed for %s", prepared.item.path)
                stats["ocr_failed"] += 1
                return ""  # не кешируем: при следующем сборе попробуем снова
        stats["ocr_calls"] += 1
        ocr_cache.put(key, text)
        return text

    async def finish(prepared: PreparedQuery) -> None:
        ocr_text = await ocr_text_for(prepared)
        ocr_features = {}
        if matcher is not None and ocr_text:
            scores = await matcher.score(ocr_text, prepared.candidate_ids)
            ocr_features = {wine_id: score.features for wine_id, score in scores.items()}
        rows = build_candidate_feature_rows(
            wine_view_matches=prepared.matches.wine_view_matches,
            wine_slugs=prepared.matches.wine_slugs,
            weights=prepared.matches.weights,
            crops=prepared.crops,
            ocr_features=ocr_features,
        )
        candidate_ids = set(prepared.candidate_ids)
        item = prepared.item
        for row in rows:
            if row.wine_id not in candidate_ids:
                continue
            out_file.write(
                json.dumps(
                    {
                        "query_id": item.query_id,
                        "expected": item.expected,
                        "image": item.query_id,
                        "source": item.source,
                        "wine_id": row.wine_id,
                        "slug": row.slug,
                        "label": 1 if row.wine_id == item.expected else 0,
                        "baseline_score": row.baseline_score,
                        "features": row.features,
                        # сырой OCR — чтобы пересчитать признаки без повторных запросов к LLM
                        "ocr_text": ocr_text,
                        "ocr_source_view": prepared.ocr_source_view,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            stats["rows"] += 1
        out_file.flush()  # запрос записан целиком — при обрыве продолжим со следующего
        stats["queries"] += 1
        if stats["queries"] % args.progress_every == 0:
            elapsed = perf_counter() - started
            print(
                f"[{stats['queries']}/{len(todo)}] rows={stats['rows']} ocr_calls={stats['ocr_calls']} "
                f"ocr_cached={stats['ocr_cached']} ocr_failed={stats['ocr_failed']} "
                f"{elapsed:.0f}s ({elapsed / stats['queries']:.2f}s/запрос)"
            )

    pending: set[asyncio.Task] = set()
    try:
        for item in todo:
            query_views, selection, vectors = await asyncio.to_thread(prepare_query, item, photo_service, visual)
            if not selection.active_views:
                logger.warning("No active views for %s", item.path)
                stats["no_views"] += 1
                continue
            # тот же поиск, что в API, но без точек самого фото-запроса
            matches = await visual.search(selection, vectors, leave_one_out_filter(item))
            base_rows = sorted(
                build_candidate_feature_rows(
                    wine_view_matches=matches.wine_view_matches,
                    wine_slugs=matches.wine_slugs,
                    weights=matches.weights,
                    crops=query_views.crops,
                ),
                key=lambda row: row.baseline_score,
                reverse=True,
            )
            source_view = ocr_source_view(query_views)
            ocr_image = query_views.images.get(source_view)
            prepared = PreparedQuery(
                item=item,
                matches=matches,
                crops=query_views.crops,
                candidate_ids=[row.wine_id for row in base_rows[: args.candidate_pool]],
                ocr_source_view=source_view,
                ocr_jpeg=image_to_jpeg_bytes(ocr_image) if llm is not None and ocr_image else None,
            )
            if args.log_each_image:
                print(f"{item.query_id} views={selection.active_views} counts={matches.per_view_counts} "
                      f"crops: {crop_status(query_views, selection.active_views)}")
            pending.add(asyncio.create_task(finish(prepared)))
            if len(pending) >= 2 * args.ocr_concurrency:  # не копим в памяти больше, чем успевает OCR
                done_tasks, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done_tasks:
                    task.result()  # не терять исключения из finish()
        if pending:
            await asyncio.gather(*pending)
    finally:
        out_file.close()
        if ocr_cache is not None:
            ocr_cache.close()
        if llm is not None:
            await llm.stop()
        if database is not None:
            await database.close()
        await qdrant.close()
        await embedder.stop()
        await bottle_cropper.stop()
        await label_cropper.stop()

    print(f"collection finished in {perf_counter() - started:.0f}s: {stats} -> {output}")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_datasets(paths: list[Path], sources: list[str] | None) -> list[dict[str, Any]]:
    records = [record for path in paths for record in load_dataset(path)]
    if sources:
        records = [record for record in records if record.get("source", "eval") in sources]
    return records


def split_by_wine(records: list[dict[str, Any]], valid_fraction: float) -> tuple[set[str], set[str]]:
    """Сплит по вину, а не по фото: фото одного вина не должны быть и в train, и в valid."""
    wines = sorted({record["expected"] for record in records})
    random.Random(42).shuffle(wines)
    valid_wines = set(wines[: max(1, int(round(len(wines) * valid_fraction)))])
    train = {r["query_id"] for r in records if r["expected"] not in valid_wines}
    valid = {r["query_id"] for r in records if r["expected"] in valid_wines}
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


def train_model(args: argparse.Namespace) -> None:
    try:
        from catboost import CatBoostRanker, Pool
    except ImportError as exc:
        raise RuntimeError("catboost is not installed. Run `uv add catboost` or `pip install catboost`.") from exc

    records = load_datasets(args.train_dataset, args.train_sources)
    if args.valid_dataset:
        valid_records = load_dataset(args.valid_dataset)
        train_records = records
    else:
        train_ids, valid_ids = split_by_wine(records, args.valid_fraction)
        train_records = [r for r in records if r["query_id"] in train_ids]
        valid_records = [r for r in records if r["query_id"] in valid_ids]
    feature_names = sorted({name for record in train_records for name in record["features"]})
    train = sorted(train_records, key=lambda record: (record["query_id"], record["wine_id"]))
    valid = sorted(valid_records, key=lambda record: (record["query_id"], record["wine_id"]))
    train_queries = {record["query_id"] for record in train}
    valid_queries = {record["query_id"] for record in valid}
    sources: dict[str, int] = {}
    for query_id, source in {(r["query_id"], r.get("source", "eval")) for r in train}:
        sources[source] = sources.get(source, 0) + 1
    print(f"train queries: {len(train_queries)} {sources}; valid queries: {len(valid_queries)}")

    def matrix(rows: list[dict[str, Any]]) -> list[list[float]]:
        return [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows]

    scale = args.stage1_baseline_scale

    def baseline(rows: list[dict[str, Any]]) -> list[float]:
        return [scale * stage1_score(row["features"]) for row in rows]

    train_pool = Pool(
        data=matrix(train),
        label=[int(row["label"]) for row in train],
        group_id=[row["query_id"] for row in train],
        feature_names=feature_names,
        baseline=baseline(train) if scale else None,
    )
    eval_set = None
    if valid:
        eval_set = Pool(
            data=matrix(valid),
            label=[int(row["label"]) for row in valid],
            group_id=[row["query_id"] for row in valid],
            feature_names=feature_names,
            baseline=baseline(valid) if scale else None,
        )

    def make_model(iterations: int) -> CatBoostRanker:
        return CatBoostRanker(
            loss_function="YetiRank",
            iterations=iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            random_seed=42,
            l2_leaf_reg=args.l2_leaf_reg,
            verbose=args.verbose_eval,
            allow_writing_files=False,
        )

    model = make_model(args.iterations)
    # early stopping по valid: при --valid-dataset data/catboost/eval.jsonl метрика на нём
    # немного оптимистична (по ней выбрано число деревьев)
    model.fit(train_pool, eval_set=eval_set, use_best_model=eval_set is not None,
              early_stopping_rounds=args.early_stopping if eval_set is not None else None)

    report_records = valid or train
    report_matrix = matrix(report_records)
    baseline_scores = [float(row["baseline_score"]) for row in report_records]
    model_scores = [
        float(score) + base
        for score, base in zip(model.predict(report_matrix), baseline(report_records), strict=True)
    ]
    print_metrics("baseline", rank_metrics(report_records, baseline_scores))
    print_metrics("stage1 formula", rank_metrics(report_records, [stage1_score(row["features"]) for row in report_records]))
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

    best_iteration = model.get_best_iteration()
    refit = bool(args.refit_with_valid and valid)
    if refit:
        # Метрики выше — честная оценка (valid не участвовал в обучении). Финальную модель
        # учим на train+valid с тем же числом деревьев: реальные фото eval тоже идут в дело.
        iterations = (best_iteration + 1) if best_iteration is not None else args.iterations
        full = sorted(train + valid, key=lambda record: (record["query_id"], record["wine_id"]))
        model = make_model(iterations)
        model.fit(
            Pool(
                data=matrix(full),
                label=[int(row["label"]) for row in full],
                group_id=[row["query_id"] for row in full],
                feature_names=feature_names,
                baseline=baseline(full) if scale else None,
            )
        )
        print(f"refit on train+valid: queries={len({r['query_id'] for r in full})} iterations={iterations}")

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
        "train_datasets": [str(path) for path in args.train_dataset],
        "train_sources": sources,
        "valid_dataset": str(args.valid_dataset) if args.valid_dataset else None,
        "best_iteration": best_iteration,
        "refit_with_valid": refit,
        "stage1_baseline_scale": scale,
    }
    args.model_output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved model: {args.model_output}")
    print(f"saved metadata: {args.model_output.with_suffix('.json')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="собрать датасет признаков (дописывает, можно продолжить)")
    collect.add_argument("--source", choices=["eval", "extended"], default="eval")
    collect.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    collect.add_argument("--extended-dir", type=Path, default=EXTENDED_DIR)
    collect.add_argument("--photo-pattern", nargs="+", default=["yandex_*", "flux_*"],
                         help="маски папок фото в images_extended (main не бывает запросом)")
    collect.add_argument("--max-per-wine", nargs="*", default=[], metavar="PATTERN=N",
                         help="не больше N фото на вино для маски, например 'flux_*=2'")
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument("--dataset-output", type=Path, default=None, help="по умолчанию data/catboost/<source>.jsonl")
    collect.add_argument("--siglip-model-dir", type=Path, default=PROJECT_ROOT / "models" / "siglip2")
    collect.add_argument("--label-model", type=Path, default=PROJECT_ROOT / "models" / "yolo" / "label.pt")
    collect.add_argument("--bottle-model", type=Path, default=PROJECT_ROOT / "models" / "yolo" / "yolo26x.pt")
    collect.add_argument("--device", default="auto")
    collect.add_argument("--candidate-pool", type=int, default=GLOBAL_CANDIDATE_LIMIT)
    collect.add_argument("--limit", type=int, default=None)
    collect.add_argument("--with-ocr", action="store_true")
    collect.add_argument("--ocr-cache", type=Path, default=DEFAULT_OCR_CACHE)
    collect.add_argument("--ocr-concurrency", type=int, default=1,
                         help="параллельных запросов к LLM (локальной модели обычно хватает 1-2)")
    collect.add_argument("--progress-every", type=int, default=50)
    collect.add_argument("--log-each-image", action="store_true")

    train = commands.add_parser("train", help="обучить CatBoost поверх формулы этапа 1")
    train.add_argument("--train-dataset", type=Path, nargs="+", default=[DATASET_DIR / "extended.jsonl"])
    train.add_argument("--train-sources", nargs="*", default=None,
                       help="взять из train только эти источники, например vivino krsk yandex (без flux)")
    train.add_argument("--valid-dataset", type=Path, default=None,
                       help="например data/catboost/eval.jsonl; без него — сплит train по винам")
    train.add_argument("--valid-fraction", type=float, default=0.2)
    train.add_argument("--refit-with-valid", action="store_true",
                       help="после оценки на valid переобучить финальную модель на train+valid "
                            "с найденным числом деревьев")
    train.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    # Дефолты по экспериментам scripts/ocr_lab: неглубокие деревья, сильная регуляризация;
    # CatBoost с нуля переобучается, поверх формулы этапа 1 — нет.
    train.add_argument("--iterations", type=int, default=1000)
    train.add_argument("--early-stopping", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=0.03)
    train.add_argument("--depth", type=int, default=4)
    train.add_argument("--l2-leaf-reg", type=float, default=10.0)
    train.add_argument(
        "--stage1-baseline-scale",
        type=float,
        default=100.0,
        help="Обучать поверх формулы этапа 1: baseline = scale * (visual + ocr_bonus). 0 — с нуля.",
    )
    train.add_argument("--verbose-eval", type=int, default=100)
    train.add_argument("--top-features", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.command == "collect":
        asyncio.run(collect_dataset(args))
    else:
        train_model(args)


if __name__ == "__main__":
    main()
