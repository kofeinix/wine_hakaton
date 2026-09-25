#!/usr/bin/env python3
"""
Grid search по весам OCR/visual rerank на eval-наборе.

Скрипт один раз получает кандидатов через extended API, затем локально
пересчитывает final score:

    final = visual_weight * visual_score + ocr_weight * ocr_score

Перебираются только OCR/visual веса с шагом 0.1. Остальные параметры
пайплайна не меняются.

Примеры:
  python scripts/grid_search_ocr_visual_weights.py --api-url http://localhost:8000 --stages global
  python scripts/grid_search_ocr_visual_weights.py --reuse-responses
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

import httpx
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services.wine_service import (
    OCR_DOMAIN_WEIGHT,
    OCR_FUZZY_TOKEN_SET_WEIGHT,
    OCR_FUZZY_WEIGHT,
    OCR_FUZZY_WRATIO_WEIGHT,
    WineService,
)
from src.ml.text_normalization import normalize_match_text

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DEFAULT_RESPONSES_PATH = PROJECT_ROOT / "data" / "ocr_visual_grid_responses.json"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "data" / "ocr_visual_grid_results.json"
SEARCH_ENDPOINT = "/api/v1/search/image/extended"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def collect_images(eval_dir: Path, limit: int | None = None) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for folder in sorted(eval_dir.iterdir()):
        if not folder.is_dir():
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                items.append((folder.name, image_path))
    return items[:limit] if limit is not None else items


async def fetch_response(
    client: httpx.AsyncClient,
    api_url: str,
    expected: str,
    image_path: Path,
    topk: int,
    stages: list[str] | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": topk}
    if stages:
        params["stages"] = stages

    mime = MIME_BY_EXT.get(image_path.suffix.lower(), "image/jpeg")
    with image_path.open("rb") as file:
        files = {"image": (image_path.name, file, mime)}
        response = await client.post(
            f"{api_url}{SEARCH_ENDPOINT}",
            files=files,
            params=params,
            timeout=180.0,
        )
    response.raise_for_status()
    payload = response.json()
    return {
        "expected": expected,
        "image": str(image_path.relative_to(PROJECT_ROOT)),
        "response": payload,
    }


async def collect_responses(
    *,
    api_url: str,
    eval_dir: Path,
    limit: int | None,
    topk: int,
    stages: list[str] | None,
) -> list[dict[str, Any]]:
    images = collect_images(eval_dir, limit)
    if not images:
        raise RuntimeError(f"No eval images found under {eval_dir}")

    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for idx, (expected, image_path) in enumerate(images, start=1):
            try:
                row = await fetch_response(client, api_url, expected, image_path, topk, stages)
            except Exception as exc:  # noqa: BLE001
                print(f"[{idx}/{len(images)}] ERROR {image_path}: {exc}")
                rows.append(
                    {
                        "expected": expected,
                        "image": str(image_path.relative_to(PROJECT_ROOT)),
                        "error": str(exc),
                    }
                )
                continue

            result = response_matches(row["response"])
            top1 = result[0].get("wine_id") if result else None
            print(
                f"[{idx}/{len(images)}] OK {image_path.name} "
                f"expected={expected[:8]} top1={str(top1 or 'N/A')[:8]} candidates={len(result)}"
            )
            rows.append(row)
    return rows


def load_color_aliases() -> dict[str, list[str]]:
    path = PROJECT_ROOT / "data" / "db" / "color_aliases.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[str]] = {}
    for row in rows:
        key = normalize_match_text(row["color"])
        if not key:
            continue
        grouped.setdefault(key, []).append(row["alias"])
    return {
        key: WineService._dedupe_text_variants([key, *aliases])
        for key, aliases in grouped.items()
    }


def response_ocr_text(response: dict[str, Any]) -> str:
    diagnostics = response.get("diagnostics") or {}
    ocr = diagnostics.get("ocr_rerank") or {}
    return str(ocr.get("normalized_text") or "")


def response_matches(response: dict[str, Any]) -> list[dict[str, Any]]:
    return response.get("result") or response.get("results") or []


def wine_object(wine: dict[str, Any]) -> SimpleNamespace:
    producer = SimpleNamespace(name=wine.get("producer") or "")
    grape_links = [
        SimpleNamespace(grape=SimpleNamespace(name=grape, aliases=[]))
        for grape in (wine.get("grapes") or [])
    ]
    return SimpleNamespace(
        producer=producer,
        name=wine.get("name") or "",
        year=wine.get("year"),
        color=wine.get("color"),
        sugar=wine.get("sugar"),
        grape_links=grape_links,
    )


def ocr_score_for_wine(
    *,
    normalized_ocr: str,
    wine: dict[str, Any],
    color_aliases_by_color: dict[str, list[str]],
) -> float:
    wine_like = wine_object(wine)
    candidate_text = WineService._wine_to_ocr_candidate_text(wine_like)
    normalized_candidate = normalize_match_text(candidate_text)
    wratio = fuzz.WRatio(normalized_ocr, normalized_candidate) if normalized_candidate else 0.0
    token_set = fuzz.token_set_ratio(normalized_ocr, normalized_candidate) if normalized_candidate else 0.0
    fuzzy_score = (
        OCR_FUZZY_WRATIO_WEIGHT * wratio
        + OCR_FUZZY_TOKEN_SET_WEIGHT * token_set
    ) / 100.0
    structured_score = WineService._structured_ocr_score(
        normalized_ocr,
        wine_like,
        color_aliases_by_color,
    )
    return OCR_DOMAIN_WEIGHT * structured_score["domain_score"] + OCR_FUZZY_WEIGHT * fuzzy_score


def build_eval_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    color_aliases_by_color = load_color_aliases()
    records: list[dict[str, Any]] = []
    missing_ocr_text = 0
    for row in rows:
        if row.get("error"):
            records.append(
                {
                    "expected": row["expected"],
                    "image": row["image"],
                    "error": row["error"],
                }
            )
            continue

        response = row.get("response") or row
        normalized_ocr = response_ocr_text(response)
        if not normalized_ocr:
            missing_ocr_text += 1
        candidates: list[dict[str, Any]] = []
        for match in response_matches(response):
            wine = match.get("wine") or {}
            ocr_score = (
                ocr_score_for_wine(
                    normalized_ocr=normalized_ocr,
                    wine=wine,
                    color_aliases_by_color=color_aliases_by_color,
                )
                if normalized_ocr and wine
                else 0.0
            )
            candidates.append(
                {
                    "wine_id": str(match["wine_id"]),
                    "visual_score": float(match.get("max_score") or match.get("score") or 0.0),
                    "ocr_score": float(ocr_score),
                }
            )

        records.append(
            {
                "expected": row["expected"],
                "image": row["image"],
                "ocr_applied": bool(normalized_ocr),
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        )
    if missing_ocr_text:
        print(
            f"WARNING: {missing_ocr_text}/{len(rows)} rows have no OCR normalized_text; "
            "their ocr_score is treated as 0. Use responses collected by this script for exact grid search.",
            file=sys.stderr,
        )
    return records


def rank_for_weights(
    candidates: list[dict[str, Any]],
    expected: str,
    visual_weight: float,
    ocr_weight: float,
) -> tuple[int | None, str | None, float | None, float | None]:
    if not candidates:
        return None, None, None, None

    ranked = sorted(
        candidates,
        key=lambda item: (
            visual_weight * item["visual_score"] + ocr_weight * item["ocr_score"]
        ),
        reverse=True,
    )
    top = ranked[0]
    top_score = visual_weight * top["visual_score"] + ocr_weight * top["ocr_score"]
    expected_score = None
    rank = None
    for idx, item in enumerate(ranked, start=1):
        score = visual_weight * item["visual_score"] + ocr_weight * item["ocr_score"]
        if item["wine_id"] == expected:
            rank = idx
            expected_score = score
            break
    return rank, top["wine_id"], top_score, expected_score


def metrics_for_weights(
    records: list[dict[str, Any]],
    visual_weight: float,
    ocr_weight: float,
) -> dict[str, Any]:
    detail_rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("error"):
            detail_rows.append(
                {
                    "expected": record["expected"],
                    "image": record["image"],
                    "rank": None,
                    "correct": False,
                    "error": record["error"],
                }
            )
            continue

        rank, top1_id, top1_score, expected_score = rank_for_weights(
            record["candidates"],
            record["expected"],
            visual_weight,
            ocr_weight,
        )
        detail_rows.append(
            {
                "expected": record["expected"],
                "image": record["image"],
                "rank": rank,
                "correct": rank == 1,
                "top1_id": top1_id,
                "top1_score": top1_score,
                "expected_score": expected_score,
                "candidate_count": record.get("candidate_count", 0),
                "ocr_applied": record.get("ocr_applied", False),
            }
        )

    valid = [row for row in detail_rows if not row.get("error")]
    n = len(valid)
    acc1 = sum(1 for row in valid if row["rank"] == 1) / n if n else 0.0
    mrr = sum(1.0 / row["rank"] for row in valid if row["rank"]) / n if n else 0.0
    recall5 = sum(1 for row in valid if row["rank"] is not None and row["rank"] <= 5) / n if n else 0.0
    recall10 = sum(1 for row in valid if row["rank"] is not None and row["rank"] <= 10) / n if n else 0.0
    missing = sum(1 for row in valid if row["rank"] is None)
    avg_rank = mean(row["rank"] for row in valid if row["rank"]) if any(row["rank"] for row in valid) else None
    return {
        "visual_weight": visual_weight,
        "ocr_weight": ocr_weight,
        "n": n,
        "acc1": acc1,
        "mrr": mrr,
        "recall5": recall5,
        "recall10": recall10,
        "missing": missing,
        "avg_rank": avg_rank,
        "rows": detail_rows,
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\nvisual  OCR    Acc@1    MRR      R@5      R@10     missing  avg_rank")
    print("-" * 78)
    for item in results:
        avg_rank = item["avg_rank"]
        avg_rank_text = f"{avg_rank:.2f}" if avg_rank is not None else "-"
        print(
            f"{item['visual_weight']:<7.1f} {item['ocr_weight']:<6.1f} "
            f"{item['acc1']:<8.2%} {item['mrr']:<8.4f} "
            f"{item['recall5']:<8.2%} {item['recall10']:<8.2%} "
            f"{item['missing']:<8} {avg_rank_text}"
        )

    best = sorted(results, key=lambda item: (item["acc1"], item["mrr"], item["recall5"]), reverse=True)[:5]
    print("\nTop-5 by Acc@1/MRR/R@5:")
    for item in best:
        print(
            f"visual={item['visual_weight']:.1f} ocr={item['ocr_weight']:.1f} "
            f"Acc@1={item['acc1']:.2%} MRR={item['mrr']:.4f} "
            f"R@5={item['recall5']:.2%} missing={item['missing']}"
        )


async def main_async(args: argparse.Namespace) -> None:
    if args.reuse_responses:
        rows = json.loads(args.responses_path.read_text(encoding="utf-8"))
        print(f"Loaded API responses: {args.responses_path} ({len(rows)} rows)")
    else:
        rows = await collect_responses(
            api_url=args.api_url.rstrip("/"),
            eval_dir=args.eval_dir,
            limit=args.limit,
            topk=args.topk,
            stages=args.stages,
        )
        args.responses_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved API responses: {args.responses_path}")

    records = build_eval_records(rows)
    results = [
        metrics_for_weights(records, visual_weight=round(i / 10, 1), ocr_weight=round(1 - i / 10, 1))
        for i in range(11)
    ]
    print_summary(results)

    payload = {
        "step": 0.1,
        "topk": args.topk,
        "stages": args.stages,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved grid results: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search OCR/visual weights on eval images.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base API URL.")
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Limit eval images for debugging.")
    parser.add_argument("--topk", type=int, default=50, help="Candidates to request from API.")
    parser.add_argument(
        "--stages",
        action="append",
        choices=["global", "patches", "llm"],
        help="API stage. Repeat for multiple stages. For no patches use --stages global.",
    )
    parser.add_argument("--reuse-responses", action="store_true", help="Reuse saved API responses.")
    parser.add_argument("--responses-path", type=Path, default=DEFAULT_RESPONSES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
