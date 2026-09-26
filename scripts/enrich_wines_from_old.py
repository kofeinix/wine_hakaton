import argparse
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5


DEFAULT_DB_DIR = Path("data/db")
FOOD_NAMESPACE = uuid5(NAMESPACE_URL, "wine-hakaton/food")
WINE_FOOD_NAMESPACE = uuid5(NAMESPACE_URL, "wine-hakaton/wine-food")


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, rows: list[dict]) -> None:
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def split_food_pairing(value: object) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for part in text.split(","):
        item = " ".join(part.strip().split())
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def food_id(name: str) -> str:
    return str(uuid5(FOOD_NAMESPACE, name.casefold()))


def wine_food_id(wine_id: str, food_id_: str) -> str:
    return str(uuid5(WINE_FOOD_NAMESPACE, f"{wine_id}:{food_id_}"))


def enrich(db_dir: Path, dry_run: bool) -> None:
    wines_path = db_dir / "wines.json"
    old_path = db_dir / "wines_old.json"
    food_path = db_dir / "food.json"
    wine_food_path = db_dir / "wine_food.json"

    wines = load_json(wines_path)
    old_wines = load_json(old_path)
    old_by_url = {row["url"]: row for row in old_wines if normalize_text(row.get("url"))}

    foods_by_key: dict[str, dict] = {}
    wine_food_rows: list[dict] = []
    matched = 0

    for wine in wines:
        old = old_by_url.get(wine.get("source_url"))
        if old is None:
            wine.setdefault("serving_temperature", None)
            wine.setdefault("shade", None)
            continue

        matched += 1
        wine["serving_temperature"] = normalize_text(old.get("serving_temperature"))
        wine["shade"] = normalize_text(old.get("shade"))

        for name in split_food_pairing(old.get("food_pairing")):
            key = name.casefold()
            food = foods_by_key.setdefault(key, {"id": food_id(name), "name": name})
            wine_food_rows.append(
                {
                    "id": wine_food_id(wine["id"], food["id"]),
                    "wine_id": wine["id"],
                    "food_id": food["id"],
                }
            )

    food_rows = sorted(foods_by_key.values(), key=lambda row: row["name"].casefold())
    wine_food_rows = sorted(wine_food_rows, key=lambda row: (row["wine_id"], row["food_id"]))

    print(f"Matched {matched}/{len(wines)} wines by source_url/url")
    print(f"Generated {len(food_rows)} unique food rows")
    print(f"Generated {len(wine_food_rows)} wine_food rows")

    if not dry_run:
        dump_json(wines_path, wines)
        dump_json(food_path, food_rows)
        dump_json(wine_food_path, wine_food_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich wines.json from wines_old.json by source_url/url and generate food tables."
    )
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enrich(args.db_dir, args.dry_run)
