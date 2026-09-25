"""Офлайн-датасет для экспериментов с OCR: сохранённые ответы API + карточки вин из data/db.

Ответы API сохраняет scripts/eval_search.py (по умолчанию data/eval_responses_<eval-папка>.json).
Каждая запись: ожидаемый wine_id, сырой OCR-текст и кандидаты с визуальным скором ДО OCR-реранка.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services.wine_service import SUGAR_VARIANTS  # noqa: E402
from src.ml.ocr_matching import OcrCandidate, OcrCatalog  # noqa: E402
from src.ml.text_normalization import normalize_match_text as norm  # noqa: E402

DEFAULT_RESPONSES = PROJECT_ROOT / "data" / "ocr_visual_grid_responses.json"
CACHE_DIR = PROJECT_ROOT / "data" / "ocr_lab_cache"
# Эксперименты (lab_*.py) берут датасет отсюда: OCR_LAB_RESPONSES=data/eval_responses_eval.json
RESPONSES = Path(os.environ.get("OCR_LAB_RESPONSES", DEFAULT_RESPONSES))


def load_catalog() -> dict:
    db = PROJECT_ROOT / "data" / "db"

    def load(name: str):
        return json.loads((db / f"{name}.json").read_text(encoding="utf-8"))

    producers = {p["id"]: p["name"] for p in load("producers")}
    grapes: dict[str, list[str]] = {g["id"]: [g["name"]] for g in load("grapes")}
    for alias in load("grape_aliases"):
        if alias.get("alias"):
            grapes[alias["grape_id"]].append(alias["alias"])
    wine_grapes: dict[str, list[str]] = {}
    for link in load("wine_grapes"):
        ids = wine_grapes.setdefault(link["wine_id"], [])
        if link["grape_id"] not in ids:
            ids.append(link["grape_id"])
    wines = {
        w["id"]: {
            "name": w["name"],
            "producer": producers.get(w.get("producer_id")),
            "grape_ids": wine_grapes.get(w["id"], []),
            "color": w.get("color"),
            "sugar": w.get("sugar"),
            "alcohol": w.get("alcohol"),
        }
        for w in load("wines")
    }
    color_aliases: dict[str, list[str]] = {}
    for row in load("color_aliases"):
        color_aliases.setdefault(norm(row["color"]), [row["color"]]).append(row["alias"])
    return {"wines": wines, "grapes": grapes, "producers": producers, "color_aliases": color_aliases}


def build_ocr_catalog(cat: dict) -> OcrCatalog:
    """Тот же OcrCatalog, что строит API из БД, но из data/db/*.json."""
    return OcrCatalog.build(
        wine_names=[w["name"] for w in cat["wines"].values()],
        producer_names=list(cat["producers"].values()),
        grapes=cat["grapes"],
        color_variants=cat["color_aliases"],
        sugar_variants=SUGAR_VARIANTS,
    )


def ocr_candidate(cat: dict, wine_id: str) -> OcrCandidate:
    w = cat["wines"].get(wine_id)
    if w is None:
        return OcrCandidate(wine_id, None, None, {}, None, None, None)
    return OcrCandidate(
        wine_id=wine_id,
        name=w["name"],
        producer=w["producer"],
        grapes={gid: tuple(cat["grapes"][gid]) for gid in w["grape_ids"]},
        color=w["color"],
        sugar=w["sugar"],
        alcohol=w["alcohol"],
    )


def cache_dir(responses: Path) -> Path:
    stat = responses.stat()
    key = hashlib.md5(f"{responses.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:8]
    path = CACHE_DIR / f"{responses.stem}_{key}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_records(responses: Path | None = None) -> list[dict]:
    responses = responses or RESPONSES
    cache = cache_dir(responses) / "records.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    rows = json.loads(responses.read_text(encoding="utf-8"))
    records = []
    skipped = 0
    for row in rows:
        resp = row.get("response") or {}
        ocr = (resp.get("diagnostics") or {}).get("ocr_rerank") or {}
        pool = ocr.get("pool")
        if pool:  # новые ответы API: визуальный скор до OCR-реранка для всего пула
            cands = [{"wine_id": str(c["wine_id"]), "visual": float(c["visual_score"])} for c in pool]
        else:  # старый дамп: max_score как приближение визуального скора
            cands = [
                {"wine_id": str(m["wine_id"]), "visual": float(m["max_score"])}
                for m in resp.get("results") or resp.get("result") or []
            ]
        if not cands:
            skipped += 1
            continue
        records.append(
            {
                "expected": row["expected"],
                "image": row["image"],
                "raw_ocr": ocr.get("text") or "",
                "ocr": ocr.get("normalized_text") or norm(ocr.get("text") or ""),
                "cands": cands,
            }
        )
    if skipped:
        print(f"skipped {skipped} responses without candidates")
    cache.write_bytes(pickle.dumps(records))
    return records


def lab_cache() -> Path:
    """Папка для промежуточных результатов экспериментов по текущему датасету."""
    return cache_dir(RESPONSES)
