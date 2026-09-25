#!/usr/bin/env python3
"""
Оценка качества поиска вин по изображению.

Для каждой папки в data/eval (имя папки = uuid вина из БД) берутся все
изображения внутри, для каждого выполняется POST /api/v1/search/image/extended,
и полученный top-1 wine_id сравнивается с ожидаемым uuid папки.

Вывод:
  - таблица по каждой картинке: ожидаемый uuid, найденный top-1, score,
    ближайшие конкуренты (top-2..top-5) с их score;
  - сводная таблица по винам (папкам);
  - итоговые метрики качества (Accuracy@1, MRR, средние score).

Использование:
  python scripts/eval_search.py [--api-url http://localhost:8000] [--limit N] [--stages global|global,llm]
  python scripts/eval_search.py --image data/eval/<wine_id>/vivino_1.jpg --stages global
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
SEARCH_ENDPOINT = "/api/v1/search/image/extended"
COMPACT_SEARCH_ENDPOINT = "/api/v1/search/image"
CATBOOST_SEARCH_ENDPOINT = "/api/v1/search/image/catboost"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Расширение -> MIME-тип (сервер требует content_type, начинающийся с "image/")
MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def collect_images(eval_dir: Path, image_name: str | None = None) -> list[tuple[str, Path]]:
    """Возвращает список (ожидаемый_uuid, путь_к_картинке).

    Поддерживает две структуры:
      - <wine_id>/<image>.jpg            (data/eval)
      - <wine_id>/main/<image>.jpg       (data/images_extended)

    image_name: если задан, берутся только файлы с таким именем
    (например "original.jpg" для структуры <wine_id>/main/original.jpg).
    """
    items: list[tuple[str, Path]] = []
    if not eval_dir.is_dir():
        print(f"Ошибка: директория {eval_dir} не найдена", file=sys.stderr)
        return items
    for folder in sorted(eval_dir.iterdir()):
        if not folder.is_dir():
            continue
        wine_id = folder.name
        # Структура <wine_id>/main/<image>.jpg
        main_dir = folder / "main"
        if main_dir.is_dir():
            for img in sorted(main_dir.iterdir()):
                if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                    if image_name is None or img.name == image_name:
                        items.append((wine_id, img))
            continue
        # Плоская структура <wine_id>/<image>.jpg
        for img in sorted(folder.iterdir()):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                if image_name is None or img.name == image_name:
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
    catboost: bool = False,
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
        if catboost:
            endpoint = CATBOOST_SEARCH_ENDPOINT
        else:
            endpoint = COMPACT_SEARCH_ENDPOINT if views else SEARCH_ENDPOINT
        resp = await client.post(
            f"{api_url}{endpoint}",
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
    image_path: Path | None = None,
    expected_id: str | None = None,
    views: list[str] | None = None,
    topk: int = 10,
    stages: list[str] | None = None,
    main_photos_only: bool = False,
    catboost: bool = False,
    eval_dir: Path | None = None,
    image_name: str | None = None,
    responses_output: Path | None = None,
) -> None:
    if image_path is not None:
        image_path = image_path.expanduser().resolve()
        if not image_path.is_file():
            print(f"Ошибка: файл {image_path} не найден", file=sys.stderr)
            sys.exit(1)
        if image_path.suffix.lower() not in IMAGE_EXTS:
            print(f"Ошибка: неподдерживаемый формат изображения {image_path.suffix}", file=sys.stderr)
            sys.exit(1)
        # <wine_id>/<image>.jpg или <wine_id>/main/<image>.jpg
        wine_dir = image_path.parent.parent if image_path.parent.name == "main" else image_path.parent
        images = [(expected_id or wine_dir.name, image_path)]
    else:
        images = collect_images(eval_dir or EVAL_DIR, image_name=image_name)
    if not images:
        print("Нет изображений для оценки.", file=sys.stderr)
        sys.exit(1)

    if limit is not None and image_path is None:
        images = images[:limit]

    print(f"Всего изображений для оценки: {len(images)}")
    endpoint = CATBOOST_SEARCH_ENDPOINT if catboost else (COMPACT_SEARCH_ENDPOINT if views else SEARCH_ENDPOINT)
    print(f"API: {api_url}{endpoint}")
    print(f"Views: {views or 'runtime active views'}")
    print(f"Stages: {'catboost' if catboost else stages or ['global']}")
    print(f"Main photos only: {main_photos_only}")
    print(f"Top-K: {topk}\n")

    rows: list[dict] = []
    # Полные ответы API (кандидаты, OCR-текст, пул до реранка) — датасет для scripts/ocr_lab.
    responses: list[dict] = []

    def save_responses() -> None:
        if responses_output is not None and responses:
            responses_output.parent.mkdir(parents=True, exist_ok=True)
            responses_output.write_text(json.dumps(responses, ensure_ascii=False), encoding="utf-8")

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
                    catboost=catboost,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[{idx}/{len(images)}] ОШИБКА {img_path.name}: {exc}")
                rows.append(
                    {
                        "expected": expected_id,
                        "image": img_path.name,
                        "stages": ["catboost"] if catboost else stages or ["global"],
                        "main_photos_only": main_photos_only,
                        "error": str(exc),
                    }
                )
                continue

            responses.append({"expected": expected_id, "image": str(img_path), "response": data})
            if len(responses) % 50 == 0:
                save_responses()  # чекпоинт: длинный прогон не пропадёт при обрыве

            result = data.get("result") or data.get("results") or []
            diagnostics = data.get("diagnostics") or {}
            global_diag = diagnostics.get("global") or {}
            ocr_diag = diagnostics.get("ocr_rerank") or {}
            catboost_diag = diagnostics.get("catboost") or {}
            fusion = global_diag.get("fusion")
            if catboost:
                # catboost endpoint кладёт fusion в global, а OCR-статус в catboost.ocr_applied
                ocr_applied = catboost_diag.get("ocr_applied", ocr_diag.get("applied"))
                ocr_source_view = catboost_diag.get("ocr_source_view", ocr_diag.get("source_view"))
            else:
                ocr_applied = ocr_diag.get("applied")
                ocr_source_view = ocr_diag.get("source_view")
            top1 = result[0] if result else None
            top2 = result[1] if len(result) > 1 else None
            rank = rank_of_expected(result, expected_id)
            correct = rank == 1
            gap = (top1.get("score", 0.0) - top2.get("score", 0.0)) if top1 and top2 else 0.0

            rows.append(
                {
                    "expected": expected_id,
                    "image": img_path.name,
                    "stages": ["catboost"] if catboost else stages or ["global"],
                    "main_photos_only": main_photos_only,
                    "active_views": global_diag.get("active_views"),
                    "fusion": fusion,
                    "ocr_source_view": ocr_source_view,
                    "ocr_applied": ocr_applied,
                    "ocr_rerank_applied": ocr_diag.get("rerank_applied"),
                    "catboost_loaded": catboost_diag.get("loaded"),
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
            fusion_label = fusion or "n/a"
            ocr_label = f"{ocr_source_view or 'n/a'}{'*' if ocr_applied else ''}"
            print(
                f"[{idx}/{len(images)}] {status} {img_path.name} "
                f"expected={expected_id[:8]} top1={top1.get('wine_id', 'N/A')[:8] if top1 else 'N/A'} "
                f"fused={top1.get('score', 0.0) if top1 else 0.0:.4f} "
                f"raw_cos={top1.get('cosine_score', 0.0) if top1 else 0.0:.4f} "
                f"fused_gap={gap:.4f} rank={rank} "
                f"views={','.join(global_diag.get('active_views') or []) or 'n/a'} "
                f"fusion={fusion_label} ocr={ocr_label}"
            )

    print("\n" + "=" * 100)
    print("ТАБЛИЦА ПО КАРТИНКАМ")
    print("=" * 100)
    header = (
        f"{'Ожидаемый uuid':<38} {'Картинка':<14} {'Top-1 uuid':<38} "
        f"{'Fused':<8} {'RawCos':<8} {'Top-2':<8} {'Gap':<8} {'Ранг':<5} {'Результат':<6} "
        "Views Fusion OCR Конкуренты (uuid:fused)"
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
        ocr_label = f"{row.get('ocr_source_view') or '-'}{'*' if row.get('ocr_applied') else ''}"
        print(
            f"{row['expected']:<38} {row['image']:<14} "
            f"{str(row['top1_id']):<38} {row['top1_score'] or 0.0:<8.4f} "
            f"{row['top1_cos'] or 0.0:<8.4f} "
            f"{row['top2_score'] or 0.0:<8.4f} {row['gap'] or 0.0:<8.4f} "
            f"{str(row['rank']):<5} {'OK' if row['correct'] else 'MISS':<6} "
            f"{','.join(row.get('active_views') or []) or '-'} "
            f"{row.get('fusion') or '-'} {ocr_label} {competitors}"
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
    save_responses()
    if responses_output is not None:
        print(f"Ответы API (для scripts/ocr_lab) сохранены в {responses_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка качества поиска вин по изображению")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Базовый URL API")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help=(
            "Путь к папке с тестовыми изображениями. Поддерживает структуры "
            "<wine_id>/<image>.jpg и <wine_id>/main/<image>.jpg. "
            "По умолчанию data/eval."
        ),
    )
    parser.add_argument(
        "--image-name",
        default=None,
        help=(
            "Брать только файлы с таким именем (например original.jpg для "
            "структуры <wine_id>/main/original.jpg). По умолчанию все изображения."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить число обрабатываемых изображений (для отладки)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help=(
            "Путь к одному изображению для точечной проверки. "
            "Если --expected-id не указан, ожидаемый wine_id берется из имени родительской папки "
            "(для <wine_id>/main/<image> — из папки над main)."
        ),
    )
    parser.add_argument(
        "--expected-id",
        default=None,
        help="Ожидаемый wine_id для --image, если его нельзя взять из имени родительской папки.",
    )
    parser.add_argument(
        "--view",
        action="append",
        choices=["original", "bottle_crop", "label_crop"],
        help=(
            "Ограничить агрегацию одним сигналом. Можно указать несколько раз. "
            "При использовании --view скрипт дергает compact endpoint без diagnostics. "
            "По умолчанию используется полный runtime pipeline через extended endpoint."
        ),
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Максимальное число результатов, запрашиваемых у API (для Recall@k). По умолчанию 50.",
    )
    parser.add_argument(
        "--stages",
        action="append",
        default=None,
        help=(
            "Стадии pipeline: global (визуальный поиск + OCR-реранк, всегда), "
            "llm (опциональный выбор кандидата vision-LLM). По умолчанию global."
        ),
    )
    parser.add_argument(
        "--main-photos-only",
        action="store_true",
        help="Искать только по векторам main-фото, исключая yandex_* фото.",
    )
    parser.add_argument(
        "--catboost",
        action="store_true",
        help="Оценивать экспериментальный CatBoost endpoint /search/image/catboost.",
    )
    parser.add_argument(
        "--responses-output",
        type=Path,
        default=None,
        help=(
            "Куда сохранить полные ответы API для офлайн-анализа OCR (scripts/ocr_lab). "
            "По умолчанию data/eval_responses_<имя eval-папки>.json; --no-save-responses — не сохранять."
        ),
    )
    parser.add_argument("--no-save-responses", action="store_true")
    args = parser.parse_args()
    stages = normalize_stages_arg(args.stages)
    responses_output = None
    if not args.no_save_responses:
        responses_output = args.responses_output or (
            EVAL_DIR.parent / f"eval_responses_{(args.eval_dir or EVAL_DIR).name}.json"
        )
    asyncio.run(
        run(
            args.api_url,
            args.limit,
            image_path=args.image,
            expected_id=args.expected_id,
            views=args.view,
            topk=args.topk,
            stages=stages,
            main_photos_only=args.main_photos_only,
            catboost=args.catboost,
            eval_dir=args.eval_dir,
            image_name=args.image_name,
            responses_output=responses_output,
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
