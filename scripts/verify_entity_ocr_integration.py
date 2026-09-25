#!/usr/bin/env python3
"""Проверка интеграции entity-based OCR в продовый код.

Считает OCR-скор через WineService._score_ocr_candidate (та же формула, что
в rerank и в признаках CatBoost) на сохранённых ответах API
(data/ocr_visual_grid_responses.json) и перебирает веса visual/ocr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services.wine_service import WineService  # noqa: E402
from src.ml.text_normalization import normalize_match_text  # noqa: E402

RESPONSES = PROJECT_ROOT / "data" / "ocr_visual_grid_responses.json"
COLOR_ALIASES = PROJECT_ROOT / "data" / "db" / "color_aliases.json"


def load_color_aliases() -> dict[str, list[str]]:
    raw = json.loads(COLOR_ALIASES.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for item in raw:
        key = normalize_match_text(item["color"])
        result.setdefault(key, [item["color"]])
        result[key].append(item["alias"])
    return result


def wine_like(wine: dict) -> SimpleNamespace:
    return SimpleNamespace(
        producer=SimpleNamespace(name=wine.get("producer") or ""),
        name=wine.get("name") or "",
        year=wine.get("year"),
        color=wine.get("color"),
        sugar=wine.get("sugar"),
        grape_links=[
            SimpleNamespace(grape=SimpleNamespace(name=g, aliases=[]))
            for g in (wine.get("grapes") or [])
        ],
    )


def score_candidate(ocr: str, wine: dict, color_aliases) -> tuple[float, float, float]:
    """(baseline, entity, hybrid) — через ту же функцию, что и в проде."""
    wl = wine_like(wine)
    candidate_text = normalize_match_text(WineService._wine_to_ocr_candidate_text(wl))
    scores = WineService._score_ocr_candidate(ocr, candidate_text, wl, color_aliases)
    return scores.baseline_score, scores.entity.domain_score, scores.ocr_score


def build_records(color_aliases):
    rows = json.loads(RESPONSES.read_text(encoding="utf-8"))
    records = []
    for row in rows:
        resp = row.get("response") or {}
        diag = resp.get("diagnostics") or {}
        ocr = str((diag.get("ocr_rerank") or {}).get("normalized_text") or "")
        matches = resp.get("result") or resp.get("results") or []
        cands = []
        for match in matches:
            wine = match.get("wine") or {}
            if not wine:
                continue
            visual = float(match.get("max_score") or match.get("score") or 0.0)
            if ocr:
                b, e, h = score_candidate(ocr, wine, color_aliases)
            else:
                b = e = h = 0.0
            cands.append({"wine_id": str(match["wine_id"]), "visual": visual, "base": b, "entity": e, "hybrid": h})
        records.append({"expected": row["expected"], "candidates": cands})
    return records


def evaluate(records, key, vw, ow):
    ranks = []
    for rec in records:
        if not rec["candidates"]:
            ranks.append(None)
            continue
        ranked = sorted(rec["candidates"], key=lambda c: vw * c["visual"] + ow * c[key], reverse=True)
        rank = next((i for i, c in enumerate(ranked, 1) if c["wine_id"] == rec["expected"]), None)
        ranks.append(rank)
    n = len(ranks)
    return {
        "acc1": sum(1 for r in ranks if r == 1) / n,
        "mrr": sum(1.0 / r for r in ranks if r) / n,
        "r5": sum(1 for r in ranks if r and r <= 5) / n,
        "r10": sum(1 for r in ranks if r and r <= 10) / n,
        "avg_rank": mean(r for r in ranks if r) if any(ranks) else None,
    }


def main() -> None:
    ca = load_color_aliases()
    records = build_records(ca)
    print(f"records={len(records)}")
    for key, title in (
        ("base", "A) BASELINE"),
        ("entity", "B) ENTITY"),
        ("hybrid", "C) HYBRID (production formula)"),
    ):
        print(f"\n=== {title} ===")
        print(f"{'vis':>4} {'ocr':>4} {'Acc@1':>8} {'MRR':>8} {'R@5':>8} {'R@10':>8} {'avg_rank':>9}")
        best = None
        for i in range(11):
            vw, ow = round(i / 10, 1), round(1 - i / 10, 1)
            m = evaluate(records, key, vw, ow)
            print(f"{vw:>4.1f} {ow:>4.1f} {m['acc1']:>8.2%} {m['mrr']:>8.4f} {m['r5']:>8.2%} {m['r10']:>8.2%} {m['avg_rank']:>9.3f}")
            if best is None or m["acc1"] > best[0]:
                best = (m["acc1"], vw, ow, m)
        print(f"BEST: visual={best[1]} ocr={best[2]} Acc@1={best[0]:.2%} MRR={best[3]['mrr']:.4f}")


if __name__ == "__main__":
    main()