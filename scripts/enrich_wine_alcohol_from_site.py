import argparse
import json
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_DB_DIR = Path("data/db")


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def slug_from_source_url(source_url: str | None) -> str | None:
    if not source_url:
        return None
    parts = urlparse(source_url).path.strip("/").split("/")
    if len(parts) == 2 and parts[0] == "wines" and parts[1]:
        return parts[1]
    return None


def format_number(value: object) -> str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return f"{number:g}"


def format_alcohol(payload: dict) -> str | None:
    alcohol = format_number(payload.get("alcohol"))
    alcohol_max = format_number(payload.get("alcoholMax"))
    if alcohol and alcohol_max and alcohol != alcohol_max:
        return f"{alcohol}-{alcohol_max}%"
    if alcohol:
        return f"{alcohol}%"
    if alcohol_max:
        return f"{alcohol_max}%"
    return None


def fetch_alcohol(slug: str, timeout: float) -> str | None:
    url = f"https://vino-svoe.ru/api/wines/{slug}"
    request = urllib.request.Request(url, headers={"User-Agent": "WineHakaton data enrichment"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return format_alcohol(payload)


def enrich(db_dir: Path, delay: float, timeout: float, limit: int | None, dry_run: bool) -> None:
    wines_path = db_dir / "wines.json"
    wines = load_json(wines_path)
    targets = [wine for wine in wines if wine.get("alcohol") is None]
    if limit is not None:
        targets = targets[:limit]

    updated = 0
    failed = 0
    for index, wine in enumerate(targets, start=1):
        slug = slug_from_source_url(wine.get("source_url"))
        if slug is None:
            failed += 1
            print(f"{index}/{len(targets)} skip bad source_url: {wine.get('source_url')}")
            continue

        try:
            alcohol = fetch_alcohol(slug, timeout=timeout)
        except Exception as exc:
            failed += 1
            print(f"{index}/{len(targets)} failed {slug}: {type(exc).__name__}: {exc}")
        else:
            print(f"{index}/{len(targets)} {slug}: {alcohol}")
            if alcohol:
                wine["alcohol"] = alcohol
                updated += 1
                if not dry_run:
                    dump_json(wines_path, wines)

        if index < len(targets):
            time.sleep(delay)

    print(f"Updated {updated}; failed/skipped {failed}; remaining null {sum(w.get('alcohol') is None for w in wines)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Carefully enrich missing wine alcohol values from vino-svoe.ru.")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to wait between requests.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout in seconds.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of missing wines to try.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enrich(
        db_dir=args.db_dir,
        delay=args.delay,
        timeout=args.timeout,
        limit=args.limit,
        dry_run=args.dry_run,
    )
