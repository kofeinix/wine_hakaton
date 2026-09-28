#!/usr/bin/env python3
"""Синонимы производителей для OCR → data/db/producer_aliases.json.

На этикетке производитель часто написан латиницей по-французски или по-английски («CHÂTEAU PINOT»),
а в каталоге — по-русски («Шато Пино»). Побуквенная транслитерация даёт «shato pino» и такое
написание не находит, поэтому храним явные синонимы (как grape_aliases у сортов).

Откуда слова: из grape_aliases берутся пары «Пино Нуар» ↔ «pinot noir» и из них слова, у которых
латинское написание не совпадает с транслитерацией (пино → pinot, нуар → noir, совиньон → sauvignon);
плюс короткий ручной словарь несортовых слов (шато → chateau, домен → domaine…). Синоним
записывается, только если в имени производителя нашлось хотя бы одно такое слово.

Уже записанные в файле синонимы сохраняются (их можно править руками).

    python3 scripts/build_producer_aliases.py --dry-run
    python3 scripts/build_producer_aliases.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_ROOT / "data" / "db"

# несортовые слова, которые на этикетках пишут не по транслитерации
MANUAL_WORDS = {
    "шато": "chateau",
    "домен": "domaine",
    "мезон": "maison",
    "кюве": "cuvee",
    "резерв": "reserve",
    "гран": "grand",
    "крю": "cru",
    "блан": "blanc",
    "нуар": "noir",
    "руж": "rouge",
    "розе": "rose",
    "брют": "brut",
    "вайнери": "winery",
    "эстейт": "estate",
    "винъярдс": "vineyards",
    "фэмили": "family",
}

_RU_LAT = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
        "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
        "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
)
_WORD = re.compile(r"[a-zа-яё]+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def translit(word: str) -> str:
    return word.translate(_RU_LAT)


def word_dictionary() -> dict[str, str]:
    """Русское слово -> написание как на этикетке, из пар «сорт — латинский синоним»."""
    grapes = {g["id"]: g["name"] for g in json.loads((DB_DIR / "grapes.json").read_text(encoding="utf-8"))}
    found: dict[str, str] = {}
    for row in json.loads((DB_DIR / "grape_aliases.json").read_text(encoding="utf-8")):
        ru, lat = words(grapes.get(row["grape_id"], "")), words(row["alias"])
        if len(ru) != len(lat) or not all(re.fullmatch(r"[а-яё]+", w) for w in ru):
            continue
        for ru_word, lat_word in zip(ru, lat):
            # только если транслитерация не справилась бы сама: «пино» -> «pinot», но не «мерло» -> «merlo»
            if translit(ru_word) != lat_word and len(ru_word) > 2:
                found.setdefault(ru_word, lat_word)
    return {**found, **MANUAL_WORDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dictionary = word_dictionary()
    producers = json.loads((DB_DIR / "producers.json").read_text(encoding="utf-8"))
    path = DB_DIR / "producer_aliases.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    rows = {(row["producer_id"], row["alias"].lower()): row for row in existing}

    added = []
    for producer in producers:
        name_words = words(producer["name"])
        if not name_words or not any(w in dictionary for w in name_words):
            continue
        alias = " ".join(dictionary.get(w, translit(w)) for w in name_words).title()
        key = (producer["id"], alias.lower())
        if key not in rows:
            rows[key] = {"producer_id": producer["id"], "alias": alias}
            added.append(f"{producer['name']} → {alias}")

    print(f"словарь слов: {len(dictionary)} ({', '.join(f'{k}→{v}' for k, v in sorted(dictionary.items())[:12])}…)")
    print(f"новых синонимов: {len(added)}")
    for line in added:
        print("  ", line)
    if args.dry_run:
        print("--dry-run: файл не изменён")
        return
    result = sorted(rows.values(), key=lambda row: (row["producer_id"], row["alias"]))
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{path.relative_to(PROJECT_ROOT)}: {len(result)} синонимов")


if __name__ == "__main__":
    main()
