from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True)
class WinePatchIndex:
    rows: list[dict[str, Any]]

    @classmethod
    def load(cls, path: Path) -> "WinePatchIndex":
        return cls(rows=list(_read_jsonl(path)))


def _open_text_reader(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text_reader(path) as reader:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Invalid patch index row at line {line_number}: expected object")
            yield row
