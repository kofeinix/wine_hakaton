from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.text_normalization import normalize_match_text


DEFAULT_DB_DIR = Path("data/db")
WINE_GRAPE_LINK_NAMESPACE = uuid.UUID("9a7974d8-0918-5c32-9294-3d516c64b7f6")
EXCLUDED_GRAPE_NAMES = {
    normalize_match_text("Олег"),
}


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def token_spans(tokens: list[str], phrase_tokens: list[str]) -> list[tuple[int, int]]:
    if not phrase_tokens or len(phrase_tokens) > len(tokens):
        return []
    spans: list[tuple[int, int]] = []
    phrase_len = len(phrase_tokens)
    for start in range(0, len(tokens) - phrase_len + 1):
        if tokens[start : start + phrase_len] == phrase_tokens:
            spans.append((start, start + phrase_len))
    return spans


def overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def build_grape_matchers(
    grapes: list[dict[str, Any]],
    grape_aliases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grape_by_id = {row["id"]: row for row in grapes}
    aliases_by_grape_id: dict[str, list[str]] = {}
    for row in grape_aliases:
        aliases_by_grape_id.setdefault(row["grape_id"], []).append(row["alias"])

    matchers: list[dict[str, Any]] = []
    for grape in grapes:
        if normalize_match_text(grape["name"]) in EXCLUDED_GRAPE_NAMES:
            continue
        variants = [grape["name"], *aliases_by_grape_id.get(grape["id"], [])]
        for variant in variants:
            normalized = normalize_match_text(variant)
            phrase_tokens = normalized.split()
            if not phrase_tokens:
                continue
            matchers.append(
                {
                    "grape_id": grape["id"],
                    "grape_name": grape_by_id[grape["id"]]["name"],
                    "variant": variant,
                    "tokens": phrase_tokens,
                    "length": len(phrase_tokens),
                }
            )

    # Prefer longer phrases, e.g. Cabernet Sauvignon over Sauvignon.
    return sorted(matchers, key=lambda item: (item["length"], len(" ".join(item["tokens"]))), reverse=True)


def find_grapes_in_wine_name(
    wine_name: str,
    matchers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    name_tokens = normalize_match_text(wine_name).split()
    matches: list[dict[str, Any]] = []
    occupied_spans: list[tuple[int, int]] = []
    matched_grape_ids: set[str] = set()

    for matcher in matchers:
        if matcher["grape_id"] in matched_grape_ids:
            continue
        spans = token_spans(name_tokens, matcher["tokens"])
        for span in spans:
            if any(overlaps(span, occupied) for occupied in occupied_spans):
                continue
            occupied_spans.append(span)
            matched_grape_ids.add(matcher["grape_id"])
            matches.append(
                {
                    "grape_id": matcher["grape_id"],
                    "grape_name": matcher["grape_name"],
                    "matched_variant": matcher["variant"],
                    "span": span,
                }
            )
            break

    return sorted(matches, key=lambda item: item["span"])


def wine_grape_row(wine_id: str, grape_id: str) -> dict[str, Any]:
    link_id = uuid.uuid5(WINE_GRAPE_LINK_NAMESPACE, f"{wine_id}:{grape_id}")
    return {
        "id": str(link_id),
        "wine_id": wine_id,
        "grape_id": grape_id,
        "percentage": None,
    }


def enrich_wine_grapes(db_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wines = load_json(db_dir / "wines.json")
    grapes = load_json(db_dir / "grapes.json")
    grape_aliases = load_json(db_dir / "grape_aliases.json")
    wine_grapes = load_json(db_dir / "wine_grapes.json")

    existing_pairs = {
        (row["wine_id"], row["grape_id"])
        for row in wine_grapes
    }
    matchers = build_grape_matchers(grapes, grape_aliases)

    additions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for wine in wines:
        matches = find_grapes_in_wine_name(wine["name"], matchers)
        new_matches = [
            match
            for match in matches
            if (wine["id"], match["grape_id"]) not in existing_pairs
        ]
        if not new_matches:
            continue

        for match in new_matches:
            row = wine_grape_row(wine["id"], match["grape_id"])
            additions.append(row)
            existing_pairs.add((row["wine_id"], row["grape_id"]))

        diagnostics.append(
            {
                "wine_id": wine["id"],
                "wine_name": wine["name"],
                "added_grapes": [
                    {
                        "grape_id": match["grape_id"],
                        "grape_name": match["grape_name"],
                        "matched_variant": match["matched_variant"],
                    }
                    for match in new_matches
                ],
            }
        )

    return wine_grapes + additions, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich wine_grapes.json from known grape names found in wine names.")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--write", action="store_true", help="Write additions back to wine_grapes.json.")
    parser.add_argument("--limit-log", type=int, default=30, help="How many additions to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, diagnostics = enrich_wine_grapes(args.db_dir)
    additions_count = sum(len(item["added_grapes"]) for item in diagnostics)
    print(f"Found {additions_count} missing wine_grapes links across {len(diagnostics)} wines.")
    for item in diagnostics[: args.limit_log]:
        grapes = ", ".join(grape["grape_name"] for grape in item["added_grapes"])
        print(f"{item['wine_id']} | {item['wine_name']} -> {grapes}")

    if args.write:
        write_json(args.db_dir / "wine_grapes.json", rows)
        print(f"Wrote {len(rows)} rows to {args.db_dir / 'wine_grapes.json'}.")
    else:
        print("Dry run only. Pass --write to update wine_grapes.json.")


if __name__ == "__main__":
    main()
