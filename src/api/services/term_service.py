"""Словарь винных терминов: справка и основы слов для подсветки терминов в тексте.

Подсветка работает на фронтенде: слово текста считается упоминанием термина, если оно равно
основе термина плюс одно из грамматических окончаний (ENDINGS). Так «амфорах» находит «Амфора»,
а короткое «Кло» не находится в «клон». Окончания отдаются вместе с основами, чтобы правила
были в одном месте.
"""

from __future__ import annotations

import asyncio
import re
from uuid import UUID

from sqlalchemy import select

from src.api.schemas import WineTermDetail, WineTermIndex, WineTermsResponse
from src.connections.database.models import WineTerm
from src.connections.database.postgres import DatabaseClient

# окончания русских существительных и прилагательных (без ё — текст нормализуется в «е»)
ENDINGS = (
    "", "а", "я", "о", "е", "ы", "и", "у", "ю", "ь", "й",
    "ам", "ям", "ах", "ях", "ов", "ев", "ей", "ой", "ом", "ем", "ою", "ею",
    "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ым", "им", "ых", "их", "ую", "юю",
    "ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими",
)
_STRIP = sorted((ending for ending in ENDINGS if ending), key=len, reverse=True)
MIN_STEM = 3
# несклоняемые заимствования, совпадающие с началом русских слов: только точное совпадение
# («Балк» — не «Золотая Балка», «Корк» — не «корками»)
EXACT_ONLY = {"балк", "корк"}
WORD = re.compile(r"[a-zа-я0-9]+(?:-[a-zа-я0-9]+)*")


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


def stem(word: str) -> str:
    """Основа слова: отрезаем самое длинное окончание, если остаётся не меньше трёх букв."""
    for ending in _STRIP:
        if word.endswith(ending) and len(word) - len(ending) >= MIN_STEM:
            return word[: -len(ending)]
    return word


def term_stems(term: str) -> list[str]:
    """«Игристое вино» → ['игрист', 'вин']; пояснение в скобках не ищем: «АОС (Appellation…)» → ['аос']."""
    phrase = re.sub(r"\([^)]*\)", " ", normalize(term))
    return [stem(word) for word in WORD.findall(phrase)]


class TermService:
    def __init__(self, database: DatabaseClient) -> None:
        self.database = database
        self._index: WineTermsResponse | None = None
        self._lock = asyncio.Lock()

    async def index(self) -> WineTermsResponse:
        """Все термины с основами — словарь небольшой (~150), держим в памяти."""
        if self._index is None:
            async with self._lock:
                if self._index is None:
                    async with self.database.session() as session:
                        terms = (await session.scalars(select(WineTerm).order_by(WineTerm.term))).all()
                    self._index = WineTermsResponse(
                        endings=list(ENDINGS),
                        items=[
                            WineTermIndex(
                                id=str(term.id),
                                term=term.term,
                                letter=term.letter,
                                stems=term_stems(term.term),
                                exact=normalize(term.term) in EXACT_ONLY,
                            )
                            for term in terms
                        ],
                    )
        return self._index

    async def get(self, term_id: UUID) -> WineTermDetail | None:
        async with self.database.session() as session:
            term = await session.get(WineTerm, term_id)
        if term is None:
            return None
        return WineTermDetail(
            id=str(term.id),
            term=term.term,
            letter=term.letter,
            definition=term.definition,
            source_url=term.source_url,
        )
