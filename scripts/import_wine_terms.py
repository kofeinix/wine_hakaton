"""Словарь винных терминов → data/db/wine_terms.json (дальше его загружает init_database.py).

Источник — https://luding-group.ru/slovar-vinnykh-terminov/ : одна страница, один запрос.
Разметка: <div class="h2-heading">А</div><p><strong>Термин</strong> - определение</p>.

    python3 scripts/import_wine_terms.py              # скачать и разобрать
    python3 scripts/import_wine_terms.py --html page.html   # разобрать сохранённую страницу
"""

from __future__ import annotations

import argparse
import html
import json
import re
import urllib.request
import uuid
from html.parser import HTMLParser
from pathlib import Path

SOURCE_URL = "https://luding-group.ru/slovar-vinnykh-terminov/"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "db" / "wine_terms.json"
# стабильные id: повторный импорт обновляет те же строки
TERM_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, SOURCE_URL)


class GlossaryParser(HTMLParser):
    """Собирает пары (буква, термин, определение) из блока lexicon-items."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[dict] = []
        self.in_lexicon = False
        self.depth = 0
        self.letter = ""
        self.mode: str | None = None  # letter / paragraph
        self.in_strong = False
        self.term: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get("class", "") or ""
        if tag == "section" and "lexicon-items" in classes:
            self.in_lexicon, self.depth = True, 0
        if not self.in_lexicon:
            return
        self.depth += 1
        if tag == "div" and "h2-heading" in classes:
            self.mode, self.text = "letter", []
        elif tag == "p":
            self.mode, self.term, self.text = "paragraph", [], []
        elif tag == "strong" and self.mode == "paragraph":
            self.in_strong = True

    def handle_endtag(self, tag):
        if not self.in_lexicon:
            return
        self.depth -= 1
        if tag == "strong":
            self.in_strong = False
        elif tag == "div" and self.mode == "letter":
            self.letter = "".join(self.text).strip()
            self.mode = None
        elif tag == "p" and self.mode == "paragraph":
            self._flush()
            self.mode = None
        if tag == "section" and self.depth < 0:
            self.in_lexicon = False

    def handle_data(self, data):
        if self.mode == "paragraph" and self.in_strong:
            self.term.append(data)
        elif self.mode:
            self.text.append(data)

    def _flush(self) -> None:
        term = _clean("".join(self.term))
        definition = _clean(re.sub(r"^\s*[-–—:]\s*", "", "".join(self.text)))
        if term and definition:
            self.items.append({"letter": self.letter, "term": term, "definition": definition})


def _clean(text: str) -> str:
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\.{2,}(?=\s|$)", ".", text)  # «регионе.. Растение» на сайте


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "WineHakaton glossary import (one-time)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def parse(page: str) -> list[dict]:
    parser = GlossaryParser()
    parser.feed(page)
    rows, seen = [], set()
    for item in parser.items:
        key = item["term"].lower().replace("ё", "е")
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "id": str(uuid.uuid5(TERM_NAMESPACE, key)),
                "term": item["term"],
                "letter": item["letter"] or item["term"][:1].upper(),
                "definition": item["definition"],
                "source_url": SOURCE_URL,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--html", type=Path, help="Разобрать сохранённую страницу вместо запроса")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    page = args.html.read_text(encoding="utf-8") if args.html else fetch(SOURCE_URL)
    rows = parse(page)
    if not rows:
        raise SystemExit("Термины не найдены — поменялась разметка страницы?")
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} терминов → {args.output}")


if __name__ == "__main__":
    main()
