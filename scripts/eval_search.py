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
  python scripts/eval_search.py [--api-url http://localhost:8000] [--limit N]
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


async def search_image(client: httpx.AsyncClient, api_url: str, image_path: Path) -> dict:
    """Выполняет поиск по картинке и возвращает JSON-ответ."""
    mime = MIME_BY_EXT.get(image_path.suffix.lower(), "image/jpeg")
    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, mime)}
        resp = await client.post(
            f"{api_url}{SEARCH_ENDPOINT}",
            files=files,
            params={"limit": 10},
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


async def run(api_url: str, limit: int | None) -> None:
    images = collect_images(EVAL_DIR)
    if not images:
        print("Нет изображений для оценки.", file=sys.stderr)
        sys.exit(1)

    if limit is not None:
        images = images[:limit]

    print(f"Всего изображений для оценки: {len(images)}")
    print(f"API: {api_url}{SEARCH_ENDPOINT}\n")

    rows: list[dict] = []
    async with httpx.AsyncClient() as client:
        for idx, (expected_id, img_path) in enumerate(images, start=1):
            try:
                data = await search_image(client, api_url, img_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[{idx}/{len(images)}] ОШИБКА {img_path.name}: {exc}")
                rows.append(
                    {
                        "expected": expected_id,
                        "image": img_path.name,
                        "error": str(exc),
                    }
                )
                continue

            result = data.get("result", [])
            top1 = result[0] if result else None
            rank = rank_of_expected(result, expected_id)
            correct = rank == 1

            rows.append(
                {
                    "expected": expected_id,
                    "image": img_path.name,
                    "top1_id": top1.get("wine_id") if top1 else None,
                    "top1_score": top1.get("score") if top1 else None,
                    "rank": rank,
                    "correct": correct,
                    "result": result,
                }
            )
            status = "OK " if correct else "MISS"
            print(
                f"[{idx}/{len(images)}] {status} {img_path.name} "
                f"expected={expected_id[:8]} top1={top1.get('wine_id', 'N/A')[:8] if top1 else 'N/A'} "
                f"score={top1.get('score', 0.0) if top1 else 0.0:.4f} rank={rank}"
            )

    print("\n" + "=" * 100)
    print("ТАБЛИЦА ПО КАРТИНКАМ")
    print("=" * 100)
    header = (
        f"{'Ожидаемый uuid':<38} {'Картинка':<14} {'Top-1 uuid':<38} "
        f"{'Score':<8} {'Ранг':<5} {'Результат':<6} Конкуренты (uuid:score)"
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
        f"{'Accuracy':<9} {'Ср. score top-1':<16} {'Лучший ранг'}"
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
    # Recall@k: доля, где ожидаемое вино вообще найдено в top-10
    recall = sum(1 for r in valid if r["rank"] is not None) / n
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
    print(f"Recall@10 (вино в top-10):  {recall:.2%}")
    print(f"Средний score top-1 (все):  {avg_score_all:.4f}")
    print(f"Средний score top-1 (верно): {avg_score_correct:.4f}")
    print(f"Средний score top-1 (ошибки): {avg_score_wrong:.4f}")

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
    args = parser.parse_args()
    asyncio.run(run(args.api_url, args.limit))


if __name__ == "__main__":
    main()