#!/usr/bin/env python3
"""Продовая OCR-формула (src/ml/ocr_matching.py) на сохранённых ответах API.

Пересчитывает final = visual + ocr_bonus для каждого кандидата и печатает
Acc@1 / MRR / Recall@5 для "только изображение" и "изображение + OCR",
а также список фото, где OCR поменял top-1.

  uv run python scripts/ocr_lab/eval_formula.py
  uv run python scripts/ocr_lab/eval_formula.py --responses data/eval_responses_eval.json --show-changes
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

from lab_data import DEFAULT_RESPONSES, build_ocr_catalog, load_catalog, load_records, ocr_candidate

from src.ml.ocr_matching import OcrQuery, score_candidates

_CAT = load_catalog()
_OCR_CATALOG = build_ocr_catalog(_CAT)


def bonuses(record: dict) -> list[float]:
    if not record["ocr"]:
        return [0.0] * len(record["cands"])
    query = OcrQuery.build(record["raw_ocr"], _OCR_CATALOG)
    candidates = [ocr_candidate(_CAT, c["wine_id"]) for c in record["cands"]]
    return [score.bonus for score in score_candidates(query, _OCR_CATALOG, candidates)]


def rank(record: dict, scores: list[float]) -> int | None:
    ids = [c["wine_id"] for c in record["cands"]]
    if record["expected"] not in ids:
        return None
    expected_score = scores[ids.index(record["expected"])]
    # ничьи — пессимистично: правильный ниже равных
    return 1 + sum(1 for wid, s in zip(ids, scores) if wid != record["expected"] and s >= expected_score)


def report(title: str, ranks: list[int | None]) -> None:
    n = len(ranks)
    acc = sum(r == 1 for r in ranks) / n
    mrr = sum(1 / r for r in ranks if r) / n
    r5 = sum(1 for r in ranks if r and r <= 5) / n
    print(f"{title:28} Acc@1={acc:.2%} ({sum(r == 1 for r in ranks)}/{n})  MRR={mrr:.4f}  R@5={r5:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--show-changes", action="store_true", help="фото, где OCR поменял top-1")
    args = parser.parse_args()

    records = load_records(args.responses)
    with Pool() as pool:
        all_bonuses = pool.map(bonuses, records, chunksize=4)

    visual_ranks, final_ranks, changes = [], [], []
    for record, bonus in zip(records, all_bonuses, strict=True):
        visual = [c["visual"] for c in record["cands"]]
        final = [v + b for v, b in zip(visual, bonus)]
        visual_ranks.append(rank(record, visual))
        final_ranks.append(rank(record, final))
        if visual_ranks[-1] != final_ranks[-1] and (visual_ranks[-1] == 1 or final_ranks[-1] == 1):
            changes.append((record, visual_ranks[-1], final_ranks[-1]))

    print(f"responses: {args.responses} ({len(records)} фото)")
    report("только изображение", visual_ranks)
    report("изображение + OCR", final_ranks)
    fixed = sum(1 for _, v, f in changes if f == 1)
    broken = sum(1 for _, v, f in changes if v == 1)
    print(f"OCR исправил top-1: {fixed}, сломал: {broken}")
    if args.show_changes:
        for record, v, f in changes:
            tag = "FIXED " if f == 1 else "BROKEN"
            print(f"  {tag} rank {v} -> {f}  {record['image']}  OCR: {record['ocr'][:90]}")


if __name__ == "__main__":
    main()
