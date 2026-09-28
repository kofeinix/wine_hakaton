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


# лигатуры и буквы, которые не раскладываются в «букву + диакритику»
_LATIN_SPECIAL = str.maketrans({"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "ł": "l", "đ": "d", "ı": "i"})


def _strip_latin_diacritics(text: str) -> str:
    """château → chateau, rosé → rose. Только латиница: кириллические «й» и «ё» не трогаем."""
    out = []
    for char in text.translate(_LATIN_SPECIAL):
        if "\u00c0" <= char <= "\u024f":  # Latin-1 Supplement и Latin Extended-A/B
            char = "".join(c for c in unicodedata.normalize("NFD", char) if not unicodedata.combining(c))
        out.append(char)
    return "".join(out)


def normalize_match_text(value: object) -> str:
    text = "" if value is None else str(value)
    # диакритику снимаем до фильтра символов: иначе «CHÂTEAU» распадается на «ch teau»
    text = _strip_latin_diacritics(unicodedata.normalize("NFKC", text).lower()).replace("ё", "е")
    text = text.translate(_CONFUSABLE_TRANSLATION)
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
