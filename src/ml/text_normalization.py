from __future__ import annotations

import re
import unicodedata


_CONFUSABLE_TRANSLATION = str.maketrans(
    {
        "а": "a",
        "о": "o",
        "с": "c",
        "р": "p",
        "е": "e",
    }
)


def normalize_match_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    text = text.translate(_CONFUSABLE_TRANSLATION)
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
