"""Дозаполняет пустые поля вин из API vino-svoe.ru (по source_url из data/db/wines.json).

Бережно к сайту: запросы строго последовательно, пауза --delay секунд после каждого,
ответы кешируются в --cache-dir (повторный запуск не ходит на сайт за уже скачанным),
при 403/429/503 или серии ошибок подряд — остановка.
Заполняются только пустые поля: shade, serving_temperature, alcohol, sugar, блюда (food / wine_food).
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from enrich_wine_alcohol_from_site import format_alcohol, slug_from_source_url
from enrich_wines_from_old import dump_json, food_id, load_json, normalize_text, wine_food_id

DEFAULT_DB_DIR = Path("data/db")
DEFAULT_CACHE_DIR = Path("data/site_cache")
API_URL = "https://vino-svoe.ru/api/wines/{slug}"
USER_AGENT = "WineHakaton data enrichment"
STOP_STATUSES = {403, 429, 503}  # сайт просит притормозить — не продолжаем
# несуществующее вино API отдаёт как 500 (страница вина при этом 404)
NOT_FOUND_STATUSES = {404, 500}
KNOWN_SUGARS = ("экстра брют", "брют", "полусухое", "полусладкое", "сухое", "сладкое")
# на сайте бывают опечатки (крепость 108 вместо 10.8) — такое не пишем, а выводим в отчёт
ALCOHOL_RANGE = (3.0, 25.0)
TEMPERATURE_RANGE = (0.0, 25.0)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


class StopScraping(Exception):
    pass


def fetch_payload(slug: str, cache_dir: Path, timeout: float) -> tuple[dict | None, bool]:
    """(payload или None если вина нет на сайте, был ли сетевой запрос)."""
    cache_path = cache_dir / f"{slug}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached.get("payload"), False

    request = urllib.request.Request(API_URL.format(slug=slug), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in STOP_STATUSES:
            raise StopScraping(f"HTTP {exc.code} for {slug}") from exc
        if exc.code not in NOT_FOUND_STATUSES:
            raise
        payload = None

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"slug": slug, "payload": payload}, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload, True


def format_temperature(value: object) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    text = text.replace("–", "-").replace("°C", "").replace("°", "").strip()
    return f"{text}°C"


def plausible(text: str | None, bounds: tuple[float, float]) -> bool:
    numbers = [float(n.replace(",", ".")) for n in _NUMBER_RE.findall(text or "")]
    return bool(numbers) and all(bounds[0] <= n <= bounds[1] for n in numbers)


def parse_sugar(payload: dict) -> str | None:
    category = (normalize_text((payload.get("category") or {}).get("name")) or "").casefold()
    # «экстра брют» проверяется раньше «брют», «полусухое» раньше «сухое»
    for sugar in KNOWN_SUGARS:
        if category.endswith(sugar):
            return sugar
    return None


def dish_names(payload: dict) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for dish in payload.get("dishes") or []:
        name = normalize_text(dish.get("name"))
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            names.append(name)
    return names


def missing_fields(wine: dict, food_wines: set[str]) -> list[str]:
    fields = [key for key in ("shade", "serving_temperature", "alcohol", "sugar") if not wine.get(key)]
    if wine["id"] not in food_wines:
        fields.append("food")
    return fields


def enrich(db_dir: Path, cache_dir: Path, delay: float, timeout: float, limit: int | None, max_failures: int, dry_run: bool) -> None:
    wines_path, food_path, wine_food_path = db_dir / "wines.json", db_dir / "food.json", db_dir / "wine_food.json"
    wines = load_json(wines_path)
    foods = load_json(food_path)
    wine_foods = load_json(wine_food_path)

    foods_by_key = {row["name"].casefold(): row for row in foods}
    wine_food_ids = {row["id"] for row in wine_foods}
    food_wines = {row["wine_id"] for row in wine_foods}

    targets = [wine for wine in wines if missing_fields(wine, food_wines)]
    if limit is not None:
        targets = targets[:limit]
    print(f"Wines with missing fields: {len(targets)}")

    filled: dict[str, int] = {}
    not_found: list[str] = []
    suspicious: list[str] = []
    failures_in_row = 0

    def save() -> None:
        if dry_run:
            return
        dump_json(wines_path, wines)
        dump_json(food_path, sorted(foods_by_key.values(), key=lambda row: row["name"].casefold()))
        dump_json(wine_food_path, sorted(wine_foods, key=lambda row: (row["wine_id"], row["food_id"])))

    for index, wine in enumerate(targets, start=1):
        prefix = f"{index}/{len(targets)}"
        slug = slug_from_source_url(wine.get("source_url"))
        if slug is None:
            print(f"{prefix} skip bad source_url: {wine.get('source_url')}")
            continue

        requested = False
        try:
            payload, requested = fetch_payload(slug, cache_dir, timeout)
        except StopScraping as exc:
            print(f"{prefix} STOP: {exc}. Прогресс сохранён, повторите позже.")
            break
        except Exception as exc:
            requested = True
            failures_in_row += 1
            print(f"{prefix} failed {slug}: {type(exc).__name__}: {exc}")
            if failures_in_row >= max_failures:
                print(f"STOP: {failures_in_row} ошибок подряд. Прогресс сохранён.")
                break
        else:
            failures_in_row = 0
            if payload is None:
                not_found.append(slug)
                print(f"{prefix} {slug}: нет на сайте")
            else:
                values = {
                    "shade": normalize_text(payload.get("color")),
                    "serving_temperature": format_temperature(payload.get("temperature")),
                    "alcohol": format_alcohol(payload),
                    "sugar": parse_sugar(payload),
                }
                for key, bounds in (("alcohol", ALCOHOL_RANGE), ("serving_temperature", TEMPERATURE_RANGE)):
                    if values[key] and not plausible(values[key], bounds):
                        suspicious.append(f"{slug}: {key}={values[key]!r}")
                        values[key] = None
                changed = []
                for key, value in values.items():
                    if value and not wine.get(key):
                        wine[key] = value
                        changed.append(key)

                if wine["id"] not in food_wines:
                    for name in dish_names(payload):
                        food = foods_by_key.setdefault(name.casefold(), {"id": food_id(name), "name": name})
                        link_id = wine_food_id(wine["id"], food["id"])
                        if link_id not in wine_food_ids:
                            wine_food_ids.add(link_id)
                            wine_foods.append({"id": link_id, "wine_id": wine["id"], "food_id": food["id"]})
                            food_wines.add(wine["id"])
                    if wine["id"] in food_wines:
                        changed.append("food")

                for key in changed:
                    filled[key] = filled.get(key, 0) + 1
                still = missing_fields(wine, food_wines)
                print(f"{prefix} {slug}: +{','.join(changed) or '—'}" + (f" (нет на сайте: {','.join(still)})" if still else ""))
                if changed:
                    save()

        if requested and index < len(targets):
            time.sleep(delay)

    remaining = sum(bool(missing_fields(wine, food_wines)) for wine in wines)
    print(f"Filled: {filled}")
    print(f"Not found on site ({len(not_found)}): {', '.join(not_found)}")
    print(f"Suspicious values skipped ({len(suspicious)}): {'; '.join(suspicious)}")
    print(f"Wines still with missing fields: {remaining}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Carefully fill missing wine fields from vino-svoe.ru API.")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--delay", type=float, default=5.0, help="Пауза после каждого запроса к сайту, секунд.")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=None, help="Сколько вин обработать максимум.")
    parser.add_argument("--max-failures", type=int, default=5, help="Остановиться после стольких ошибок подряд.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enrich(args.db_dir, args.cache_dir, args.delay, args.timeout, args.limit, args.max_failures, args.dry_run)
