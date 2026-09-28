#!/usr/bin/env python3
"""Заменить заглушку «Неизвестный производитель» настоящим производителем с vino-svoe.ru.

У части вин вместо производителя стоит заглушка, хотя он виден в названии («Golubitskoe Estate
Chardonnay»). OCR находит производителя на этикетке и штрафует такое вино за «другого
производителя» — правильное вино уходит вниз выдачи.

Производитель берётся из API страницы вина (manufacturer); ответы кешируются в data/site_cache,
не чаще одного запроса в --delay секунд (как в enrich_wines_from_site.py). Вино привязывается к
существующему производителю с тем же именем (без учёта регистра) или к новому; заглушка удаляется,
если больше не используется.

    python3 scripts/enrich_producers_from_site.py --dry-run
    python3 scripts/enrich_producers_from_site.py
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from enrich_wines_from_site import DEFAULT_CACHE_DIR, StopScraping, fetch_payload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_ROOT / "data" / "db"
UNKNOWN_PRODUCER = "Неизвестный производитель"
PRODUCER_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "wine-hakaton/producer")
# у одной винодельни на сайте бывает несколько записей: берём ту, что уже есть в каталоге
# (на этикетке «GOLUBITSKOE ESTATE» — так OCR и находит производителя)
SITE_ALIASES = {"поместье голубицкое": "Golubitskoe Estate"}


_INDENTS: dict[str, int] = {}


def load(name: str) -> list[dict]:
    text = (DB_DIR / name).read_text(encoding="utf-8")
    rows = json.loads(text)
    # файлы каталога записаны с разным отступом — сохраняем как было, чтобы diff был только по делу
    _INDENTS[name] = next(
        (i for i in (1, 2, 4) if text == json.dumps(rows, ensure_ascii=False, indent=i) + "\n"), 2
    )
    return rows


def dump(name: str, rows: list[dict]) -> None:
    text = json.dumps(rows, ensure_ascii=False, indent=_INDENTS.get(name, 2)) + "\n"
    (DB_DIR / name).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--delay", type=float, default=2.0, help="Секунд между запросами к сайту")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / DEFAULT_CACHE_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что изменится")
    args = parser.parse_args()

    producers = load("producers.json")
    wines = load("wines.json")
    unknown_ids = {p["id"] for p in producers if p["name"].strip().casefold() == UNKNOWN_PRODUCER.casefold()}
    by_name = {p["name"].strip().casefold(): p for p in producers if p["id"] not in unknown_ids}
    targets = [w for w in wines if w["producer_id"] in unknown_ids]
    print(f"вин с заглушкой «{UNKNOWN_PRODUCER}»: {len(targets)}")

    changed, created, last_request = 0, [], 0.0
    for wine in targets:
        slug = (wine.get("source_url") or "").rstrip("/").rsplit("/", 1)[-1] or wine["sku"]
        if not (args.cache_dir / f"{slug}.json").exists():
            pause = last_request + args.delay - time.monotonic()
            if pause > 0:
                time.sleep(pause)
            last_request = time.monotonic()
        try:
            payload, _ = fetch_payload(slug, args.cache_dir, args.timeout)
        except StopScraping as exc:
            print(f"сайт просит притормозить ({exc}) — остановились, перезапустите позже")
            break
        name = ((payload or {}).get("manufacturer") or {}).get("name", "").strip()
        name = SITE_ALIASES.get(name.casefold(), name)
        if not name:
            print(f"  {wine['name']}: на сайте производитель не указан — оставляем")
            continue
        producer = by_name.get(name.casefold())
        if producer is None:
            producer = {"id": str(uuid.uuid5(PRODUCER_NAMESPACE, name.casefold())), "name": name}
            by_name[name.casefold()] = producer
            producers.append(producer)
            created.append(name)
        print(f"  {wine['name']}: → {producer['name']}")
        wine["producer_id"] = producer["id"]
        changed += 1

    still_used = {w["producer_id"] for w in wines}
    producers = [p for p in producers if p["id"] not in unknown_ids or p["id"] in still_used]
    print(f"итог: исправлено {changed} вин, новых производителей {len(created)} {created}")
    if args.dry_run:
        print("--dry-run: файлы не изменены")
        return
    dump("wines.json", wines)
    dump("producers.json", producers)
    print("data/db/wines.json и producers.json обновлены — загрузите: docker compose up db-init")


if __name__ == "__main__":
    main()
