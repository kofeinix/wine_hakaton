#!/usr/bin/env python3
"""
Оценка качества поиска вин по изображению.

Для каждой папки в data/eval (имя папки = uuid вина из БД) берутся все
изображения внутри, для каждого выполняется POST /api/v1/search/image,
и полученный top-1 wine_id сравнивается с ожидаемым uuid папки.

Вывод:
  - таблица по каждой картинке: ожидаемый uuid, найденный top-1, score,
    ближайшие конкуренты (top-2..top-5) с их score;
  - сводная таблица по винам (папкам);
  - итоговые метрики качества (Accuracy@1, MRR, средние score).

Использование:
  python scripts/eval_search.py [--api-url http://localhost:8000] [--limit N] [--stages global,patches,llm]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
SEARCH_ENDPOINT = "/api/v1/search/image"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Расширение -> MIME-тип (сервер требует content_type, начинающийся с "image/")
MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def collect_images(eval_dir: Path) -> list[tuple[str, Path]]:
    """Возвращает список (ожидаемый_uuid, путь_к_картинке)."""
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


async def search_image(
    client: httpx.AsyncClient,
    api_url: str,
    image_path: Path,
    views: list[str] | None = None,
    topk: int = 10,
    stages: list[str] | None = None,
    main_photos_only: bool = False,
) -> dict:
    """Выполняет поиск по картинке и возвращает JSON-ответ."""
    mime = MIME_BY_EXT.get(image_path.suffix.lower(), "image/jpeg")
    params: dict = {"limit": topk}
    if stages:
        params["stages"] = stages
    if views:
        params["views"] = views
    if main_photos_only:
        params["main_photos_only"] = True
    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, mime)}
        resp = await client.post(
            f"{api_url}{SEARCH_ENDPOINT}",
            files=files,
            params=params,
            timeout=120.0,
        )
    resp.raise_for_status()
    return resp.json()


def rank_of_expected(result: list[dict], expected_id: str) -> int | None:
    """Возвращает позицию (1-based) ожидаемого вина в результате или None."""
    for i, match in enumerate(result, start=1):
        if str(match.get("wine_id")) == expected_id:
            return i
    return None


async def run(
    api_url: str,
    limit: int | None,
    views: list[str] | None = None,
    topk: int = 10,
    stages: list[str] | None = None,
    main_photos_only: bool = False,
) -> None:
    images = collect_images(EVAL_DIR)
    if not images:
        print("Нет изображений для оценки.", file=sys.stderr)
        sys.exit(1)

    if limit is not None:
        images = images[:limit]

    print(f"Всего изображений для оценки: {len(images)}")
    print(f"API: {api_url}{SEARCH_ENDPOINT}")
    print(f"Views: {views or 'all (original + label_crop)'}")
    print(f"Stages: {stages or ['global', 'patches']}")
    print(f"Main photos only: {main_photos_only}")
    print(f"Top-K: {topk}\n")

    rows: list[dict] = []
    async with httpx.AsyncClient() as client:
        for idx, (expected_id, img_path) in enumerate(images, start=1):
            try:
                data = await search_image(
                    client,
                    api_url,
                    img_path,
                    views=views,
                    topk=topk,
                    stages=stages,
                    main_photos_only=main_photos_only,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[{idx}/{len(images)}] ОШИБКА {img_path.name}: {exc}")
                rows.append(
                    {
                        "expected": expected_id,
                        "image": img_path.name,
                        "stages": stages or ["global", "patches"],
                        "main_photos_only": main_photos_only,
                        "error": str(exc),
                    }
                )
                continue

            result = data.get("result", [])
            top1 = result[0] if result else None
            top2 = result[1] if len(result) > 1 else None
            rank = rank_of_expected(result, expected_id)
            correct = rank == 1
            gap = (top1.get("score", 0.0) - top2.get("score", 0.0)) if top1 and top2 else 0.0

            rows.append(
                {
                    "expected": expected_id,
                    "image": img_path.name,
                    "stages": stages or ["global", "patches"],
                    "main_photos_only": main_photos_only,
                    "top1_id": top1.get("wine_id") if top1 else None,
                    "top1_score": top1.get("score") if top1 else None,
                    "top1_cos": top1.get("cosine_score") if top1 else None,
                    "top2_id": top2.get("wine_id") if top2 else None,
                    "top2_score": top2.get("score") if top2 else None,
                    "top2_cos": top2.get("cosine_score") if top2 else None,
                    "gap": gap,
                    "rank": rank,
                    "correct": correct,
                    "result": result,
                }
            )
            status = "OK " if correct else "MISS"
            print(
                f"[{idx}/{len(images)}] {status} {img_path.name} "
                f"expected={expected_id[:8]} top1={top1.get('wine_id', 'N/A')[:8] if top1 else 'N/A'} "
                f"fused={top1.get('score', 0.0) if top1 else 0.0:.4f} "
                f"raw_cos={top1.get('cosine_score', 0.0) if top1 else 0.0:.4f} "
                f"fused_gap={gap:.4f} rank={rank}"
            )

    print("\n" + "=" * 100)
    print("ТАБЛИЦА ПО КАРТИНКАМ")
    print("=" * 100)
    header = (
        f"{'Ожидаемый uuid':<38} {'Картинка':<14} {'Top-1 uuid':<38} "
        f"{'Fused':<8} {'RawCos':<8} {'Top-2':<8} {'Gap':<8} {'Ранг':<5} {'Результат':<6} Конкуренты (uuid:fused)"
    )
    print(header)
    print("-" * 100)
    for row in rows:
        if "error" in row:
            print(f"{row['expected']:<38} {row['image']:<14} {'ОШИБКА':<38}")
            continue
        competitors = ", ".join(
            f"{m.get('wine_id', '?')[:8]}:{m.get('score', 0.0):.3f}"
            for m in row["result"][1:5]
        )
        print(
            f"{row['expected']:<38} {row['image']:<14} "
            f"{str(row['top1_id']):<38} {row['top1_score'] or 0.0:<8.4f} "
            f"{row['top1_cos'] or 0.0:<8.4f} "
            f"{row['top2_score'] or 0.0:<8.4f} {row['gap'] or 0.0:<8.4f} "
            f"{str(row['rank']):<5} {'OK' if row['correct'] else 'MISS':<6} {competitors}"
        )

    # Сводка по винам (папкам)
    print("\n" + "=" * 100)
    print("СВОДКА ПО ВИНАМ (папкам)")
    print("=" * 100)
    by_wine: dict[str, list[dict]] = {}
    for row in rows:
        if "error" in row:
            continue
        by_wine.setdefault(row["expected"], []).append(row)

    wine_header = (
        f"{'uuid вина':<38} {'Картинок':<9} {'Попаданий':<10} "
        f"{'Accuracy':<9} {'Ср. fused top-1':<16} {'Лучший ранг'}"
    )
    print(wine_header)
    print("-" * 100)
    for wine_id, wine_rows in sorted(by_wine.items()):
        n = len(wine_rows)
        hits = sum(1 for r in wine_rows if r["correct"])
        acc = hits / n if n else 0.0
        avg_score = sum(r["top1_score"] or 0.0 for r in wine_rows) / n if n else 0.0
        best_rank = min((r["rank"] for r in wine_rows if r["rank"]), default=None)
        print(
            f"{wine_id:<38} {n:<9} {hits:<10} {acc:<9.2%} "
            f"{avg_score:<16.4f} {best_rank}"
        )

    # Итоговые метрики
    print("\n" + "=" * 100)
    print("ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 100)
    valid = [r for r in rows if "error" not in r]
    n = len(valid)
    if n == 0:
        print("Нет валидных результатов.")
        return

    acc1 = sum(1 for r in valid if r["correct"]) / n
    # MRR: 1/rank если найден, иначе 0
    mrr = sum(1.0 / r["rank"] for r in valid if r["rank"]) / n
    avg_score_all = sum(r["top1_score"] or 0.0 for r in valid) / n
    avg_score_correct = sum(
        r["top1_score"] or 0.0 for r in valid if r["correct"]
    ) / max(1, sum(1 for r in valid if r["correct"]))
    avg_score_wrong = sum(
        r["top1_score"] or 0.0 for r in valid if not r["correct"]
    ) / max(1, sum(1 for r in valid if not r["correct"]))

    print(f"Изображений оценено:        {n}")
    print(f"Accuracy@1 (top-1 верный):  {acc1:.2%}")
    print(f"MRR (средний обратный ранг): {mrr:.4f}")
    print(f"Средний fused top-1 (все):  {avg_score_all:.4f}")
    print(f"Средний fused top-1 (верно): {avg_score_correct:.4f}")
    print(f"Средний fused top-1 (ошибки): {avg_score_wrong:.4f}")

    # Recall@k: доля, где ожидаемое вино найдено в top-k.
    print("\nRecall@k (вино в top-k):")
    for k in (5, 10, 20, 30, 40, 50):
        hits = sum(1 for r in valid if r["rank"] is not None and r["rank"] <= k)
        print(f"  Recall@{k:<3} = {hits / n:.2%}  ({hits}/{n})")

    # Сохраняем полный результат в JSON
    out_path = Path(__file__).resolve().parent.parent / "data" / "eval_results.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nПолные результаты сохранены в {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка качества поиска вин по изображению")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Базовый URL API")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить число обрабатываемых изображений (для отладки)",
    )
    parser.add_argument(
        "--view",
        action="append",
        choices=["original", "bottle_crop", "label_crop"],
        help=(
            "Ограничить агрегацию одним сигналом. Можно указать несколько раз. "
            "По умолчанию используются original и label_crop."
        ),
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Максимальное число результатов, запрашиваемых у API (для Recall@k). По умолчанию 10.",
    )
    parser.add_argument(
        "--stages",
        action="append",
        default=None,
        help=(
            "Стадии pipeline: global, patches, llm. Можно указать несколько раз "
            "или через запятую. Legacy: 1 = global, 2 = global+patches. "
            "По умолчанию global,patches."
        ),
    )
    parser.add_argument(
        "--main-photos-only",
        action="store_true",
        help="Искать только по векторам main-фото, исключая yandex_* фото.",
    )
    args = parser.parse_args()
    stages = normalize_stages_arg(args.stages)
    asyncio.run(
        run(
            args.api_url,
            args.limit,
            views=args.view,
            topk=args.topk,
            stages=stages,
            main_photos_only=args.main_photos_only,
        )
    )


def normalize_stages_arg(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    result: list[str] = []
    for value in values:
        for part in value.split(","):
            item = part.strip()
            if item:
                result.append(item)
    return result


if __name__ == "__main__":
    main()
