"""Сомелье без нейросети: подбор вина по поводу, блюду, вкусу и фильтрам каталога.

Каталог небольшой (~2 тыс. вин), поэтому он целиком загружается в память при первом запросе,
а фильтры, счётчики вариантов и ранжирование считаются на Python. Каждая причина в выдаче
взята из полей карточки (цвет, сахар, гастросочетания, описание, крепость, рейтинг), и её можно
проверить на странице вина.
"""

from __future__ import annotations

import asyncio
import difflib
import re
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from src.api.repositories.wine_repository import WineRepository
from src.api.schemas import (
    SommelierFacets,
    SommelierMatch,
    SommelierOption,
    SommelierSearchResponse,
    WineResponse,
)
from src.api.services.wine_service import wine_to_response
from src.connections.database.postgres import DatabaseClient

# --- справочники --------------------------------------------------------------------------

# группы блюд: в каталоге 34 названия, часть — почти дубли («Легкие закуски» / «Лёгкие закуски»)
DISH_GROUPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "cheese": ("Сыры", ("Сыры",)),
    "fish": ("Рыба и морепродукты", ("Рыба и морепродукты", "Морепродукты", "Устрицы", "Блюда из рыбы")),
    "meat": ("Мясо и стейки", ("Мясо и стейки",)),
    "poultry": ("Птица", ("Блюда из птицы",)),
    "grill": ("Гриль и BBQ", ("BBQ", "Овощи гриль")),
    "snacks": ("Лёгкие закуски", ("Легкие закуски", "Лёгкие закуски", "Закуски", "Брускетты")),
    "charcuterie": ("Мясное ассорти и паштеты", ("Мясное ассорти", "Паштеты")),
    "salads": ("Салаты и овощи", ("Салаты", "Свежие овощи", "Запеченные овощи")),
    "pasta": ("Паста и пицца", ("Паста", "Пицца", "Несладкая выпечка")),
    "desserts": (
        "Десерты",
        ("Выпечка и десерты", "Десерты", "Фруктово-ягодные десерты", "Мороженое", "Шоколад"),
    ),
    "fruits": ("Фрукты", ("Фрукты",)),
    "asian": ("Азиатская и острая кухня", ("Азиатская кухня", "Острое")),
    "world": (
        "Кухни мира",
        ("Кухни народов мира", "Кавказская кухня", "Средиземноморская кухня", "Русская кухня"),
    ),
    "fastfood": ("Фастфуд", ("Фастфуд",)),
}


@dataclass(frozen=True)
class Taste:
    label: str
    pattern: re.Pattern[str]
    # крепость как дополнительный признак (лёгкое — невысокая, плотное — высокая)
    alcohol: Callable[[float], bool] | None = None
    alcohol_reason: str = ""


TASTES: dict[str, Taste] = {
    "fresh": Taste("Бодрое", re.compile(r"свеж\w*|бодрящ\w*|цитрус\w*|кислотн\w*")),
    "fruity": Taste(
        "Ягоды и фрукты",
        re.compile(r"фрукт\w*|ягод\w*|персик\w*|абрикос\w*|вишн\w*|черешн\w*|слив\w*|малин\w*|смородин\w*|яблок\w*|груш\w*"),
    ),
    "floral": Taste("С цветочными нотами", re.compile(r"цветоч\w*|цветов\w*|цветы|акаци\w*|жасмин\w*|бузин\w*|лепестк\w*|\bроз[аыу]\b")),
    "aged": Taste("С выдержкой", re.compile(r"выдерж\w*|дуб\w*|бочк\w*|ванил\w*|табак\w*|\bкож[аиу]\b")),
    "full": Taste(
        "Насыщенное",
        re.compile(r"плотн\w*|насыщенн\w*|полнотел\w*|концентрир\w*|маслянист\w*"),
        alcohol=lambda value: value >= 14,
        alcohol_reason="крепость от 14%",
    ),
    "light": Taste(
        "Лёгкое, питкое",
        re.compile(r"л[её]гк\w*|элегантн\w*|изящн\w*|воздушн\w*"),
        alcohol=lambda value: value <= 11.5,
        alcohol_reason="крепость до 11,5%",
    ),
    "spicy": Taste("Со специями", re.compile(r"прян\w*|специ\w*|перц\w*|гвоздик\w*|кориц\w*")),
}

SPARKLING_PATTERN = re.compile(r"игрист|шампан|перляж|брют|петнат|pet[\s-]?nat")
SPARKLING_SUGARS = {"брют", "экстра брют"}
SWEET_SUGARS = {"сладкое", "полусладкое"}
DRY_SUGARS = {"сухое", "брют", "экстра брют"}
TYPES = {"sparkling": "Игристое", "still": "Тихое"}
SORTS = {"relevance", "rating", "name"}
MIN_OCCASION_SCORE = 2


# --- подготовленная карточка ------------------------------------------------------------------


@dataclass
class Entry:
    card: WineResponse
    name: str
    haystack: str  # название, производитель, сорта, регион — для поиска по тексту
    haystack_latin: str
    description: str
    dishes: frozenset[str]
    tastes: dict[str, str]  # вкус -> причина
    sparkling: bool
    alcohol: float | None
    color: str | None
    sugar: str | None
    region: str | None
    grapes: tuple[str, ...]
    rating: float = 0.0
    shuffle: int = 0


@dataclass
class Scored:
    entry: Entry
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


# --- поводы: правила «условие → баллы и причина» -----------------------------------------------

Rule = tuple[Callable[[Entry], bool], float, Callable[[Entry], str]]


def _dishes(entry: Entry, groups: set[str]) -> list[str]:
    return [DISH_GROUPS[group][0] for group in DISH_GROUPS if group in groups and group in entry.dishes]


def _dish_reason(labels: list[str]) -> str:
    return f"сочетается с: {', '.join(labels).lower()}"


def _dish_rule(groups: set[str], points: float) -> Rule:
    return (lambda entry: bool(entry.dishes & groups), points, lambda entry: _dish_reason(_dishes(entry, groups)))


def _taste_rule(taste: str, points: float) -> Rule:
    return (lambda entry: taste in entry.tastes, points, lambda entry: _taste_reason(entry, taste))


def _taste_reason(entry: Entry, taste: str) -> str:
    return f"{TASTES[taste].label.lower()} — {entry.tastes[taste]}"


def _style(entry: Entry) -> str:
    return " ".join(part for part in ((entry.color or "").lower(), entry.sugar or "") if part)


OCCASIONS: dict[str, tuple[str, list[Rule]]] = {
    "aperitif": (
        "Перед ужином",
        [
            (lambda e: e.sparkling, 2, lambda e: "игристое — классика аперитива"),
            (
                lambda e: not e.sparkling and e.color in {"Белое", "Розовое"} and e.sugar in DRY_SUGARS,
                1.5,
                lambda e: f"{_style(e)} — освежает перед едой",
            ),
            _dish_rule({"snacks"}, 1),
            _taste_rule("fresh", 0.5),
        ],
    ),
    "dinner": (
        "К ужину",
        [
            _dish_rule({"meat", "poultry", "fish", "pasta"}, 2),
            (lambda e: not e.sparkling and e.sugar in DRY_SUGARS, 1, lambda e: f"{_style(e)} — не перебьёт еду"),
        ],
    ),
    "gift": (
        "В подарок",
        [
            (lambda e: e.rating >= 4.7, 2, lambda e: f"народный рейтинг {e.rating:.2f}".replace(".", ",")),
            _taste_rule("aged", 1),
            (lambda e: e.sparkling, 0.5, lambda e: "игристое — к празднику"),
        ],
    ),
    "party": (
        "Для компании",
        [
            (lambda e: e.sparkling, 2, lambda e: "игристое — для компании"),
            (lambda e: e.color == "Розовое", 1, lambda e: "розовое — нравится большинству"),
            (lambda e: e.sugar in {"полусухое", "полусладкое"}, 1, lambda e: f"{e.sugar} — мягкий вкус без терпкости"),
            _dish_rule({"snacks", "pasta", "fastfood"}, 1),
        ],
    ),
    "date": (
        "Романтический вечер",
        [
            (lambda e: e.sparkling or e.color == "Розовое", 2, lambda e: "игристое" if e.sparkling else "розовое вино"),
            _dish_rule({"desserts", "fruits", "cheese"}, 1),
            _taste_rule("floral", 0.5),
            _taste_rule("fruity", 0.5),
        ],
    ),
    "picnic": (
        "На природу",
        [
            (lambda e: e.color in {"Розовое", "Белое"}, 1, lambda e: f"{(e.color or '').lower()} — хорошо охлаждённым"),
            _taste_rule("light", 1),
            _dish_rule({"salads", "snacks", "cheese"}, 1),
        ],
    ),
    "grill": (
        "К мангалу",
        [
            (
                lambda e: e.color == "Красное" and e.sugar in DRY_SUGARS,
                2,
                lambda e: "красное сухое — к жареному мясу",
            ),
            _dish_rule({"grill", "meat", "world"}, 2),
            _taste_rule("spicy", 0.5),
            _taste_rule("full", 0.5),
        ],
    ),
    "dessert": (
        "К сладкому",
        [
            (lambda e: e.sugar in SWEET_SUGARS, 2, lambda e: f"{e.sugar} — не спорит со сладким"),
            _dish_rule({"desserts", "fruits"}, 2),
        ],
    ),
}


# --- текст ------------------------------------------------------------------------------------

_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z", "и": "i", "й": "i",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e",
    "ю": "yu", "я": "ya",
}


def normalize(text: str | None) -> str:
    text = (text or "").lower().replace("ё", "е")
    return " ".join(re.findall(r"[0-9a-zа-я]+", text))


def to_latin(text: str) -> str:
    """Грубая транслитерация: «абрау» и «abrau» сравниваются в одном алфавите."""
    latin = "".join(_LATIN.get(char, char) for char in text)
    # частые варианты написания: kh/h, ks/x, w/v, j/i
    return latin.replace("kh", "h").replace("x", "ks").replace("w", "v").replace("j", "i").replace("y", "i")


def _parse_alcohol(value: str | None) -> float | None:
    numbers = [float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", value or "")]
    numbers = [n for n in numbers if 3 <= n <= 25]
    return sum(numbers) / len(numbers) if numbers else None


def _taste_reasons(description: str, alcohol: float | None) -> dict[str, str]:
    reasons = {}
    for key, taste in TASTES.items():
        match = taste.pattern.search(description)
        if match:
            reasons[key] = f"в описании: «{match.group(0)}»"
        elif taste.alcohol and alcohol is not None and taste.alcohol(alcohol):
            reasons[key] = taste.alcohol_reason
    return reasons


def _entry(card: WineResponse) -> Entry:
    description = (card.description or "").lower().replace("ё", "е")
    food_to_group = {food: group for group, (_, foods) in DISH_GROUPS.items() for food in foods}
    alcohol = _parse_alcohol(card.alcohol)
    sparkling = card.sugar in SPARKLING_SUGARS or bool(SPARKLING_PATTERN.search(f"{card.name.lower()} {description}"))
    # сахар, цвет и «игристое» тоже ищутся: «галицкий брют», «саперави красное»
    haystack = normalize(
        " ".join(
            [
                card.name,
                card.producer or "",
                *card.grapes,
                card.region or "",
                card.color or "",
                card.sugar or "",
                "игристое" if sparkling else "",
            ]
        )
    )
    return Entry(
        card=card,
        name=normalize(card.name),
        haystack=haystack,
        haystack_latin=to_latin(haystack),
        description=description,
        dishes=frozenset(food_to_group.get(food, food) for food in card.food_pairings),
        tastes=_taste_reasons(description, alcohol),
        sparkling=sparkling,
        alcohol=alcohol,
        color=card.color,
        sugar=card.sugar or None,
        region=card.region,
        grapes=tuple(card.grapes),
        rating=card.rating or 0.0,
        shuffle=zlib.crc32(card.id.encode()),
    )


# --- сервис -----------------------------------------------------------------------------------


@dataclass
class SommelierQuery:
    text: str = ""
    occasion: str | None = None
    dishes: list[str] = field(default_factory=list)
    tastes: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    sugars: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    grapes: list[str] = field(default_factory=list)
    min_rating: float | None = None
    sort: str = "relevance"
    limit: int = 12
    offset: int = 0


class SommelierService:
    def __init__(self, database: DatabaseClient) -> None:
        self.repository = WineRepository(database)
        self._entries: list[Entry] | None = None
        self._vocabulary: list[str] = []
        self._lock = asyncio.Lock()

    async def _catalog(self) -> list[Entry]:
        if self._entries is None:
            async with self._lock:
                if self._entries is None:
                    wines = await self.repository.load_all_wines()
                    entries = [_entry(wine_to_response(wine)) for wine in wines]
                    self._vocabulary = sorted({word for entry in entries for word in entry.haystack.split()})
                    self._entries = entries
        return self._entries

    async def search(self, query: SommelierQuery) -> SommelierSearchResponse:
        entries = await self._catalog()
        text = normalize(query.text)
        matched = self._run(entries, query, text)
        corrected = None
        if text and not matched:
            fixed = self._correct(text)
            if fixed and fixed != text:
                matched = self._run(entries, query, fixed)
                corrected = fixed if matched else None
        ranked = self._sort(matched, query.sort)
        page = ranked[query.offset : query.offset + query.limit]
        return SommelierSearchResponse(
            total=len(ranked),
            items=[
                SommelierMatch(wine=item.entry.card, score=round(item.score, 2), reasons=item.reasons) for item in page
            ],
            corrected_query=corrected,
            facets=self._facets(entries, query, corrected or text),
        )

    # --- отбор -------------------------------------------------------------------------------

    def _run(self, entries: list[Entry], query: SommelierQuery, text: str, skip: str | None = None) -> list[Scored]:
        """Вина, прошедшие фильтры (кроме измерения skip — для счётчиков), с баллами и причинами."""
        tokens = text.split()
        result = []
        for entry in entries:
            if not self._passes(entry, query, skip):
                continue
            scored = Scored(entry)
            if tokens:
                text_score = _text_score(entry, text, tokens)
                if not text_score:
                    continue
                scored.score += text_score
            if query.occasion in OCCASIONS and skip != "occasion":
                points, reasons = _occasion(entry, query.occasion)
                if points < MIN_OCCASION_SCORE:
                    continue
                scored.score += points
                scored.reasons += reasons
            if query.dishes:
                scored.reasons.append(_dish_reason([DISH_GROUPS[g][0] if g in DISH_GROUPS else g for g in query.dishes]))
            for taste in query.tastes:
                if taste in entry.tastes:
                    scored.reasons.append(_taste_reason(entry, taste))
            scored.reasons = list(dict.fromkeys(scored.reasons))
            result.append(scored)
        return result

    @staticmethod
    def _passes(entry: Entry, query: SommelierQuery, skip: str | None) -> bool:
        checks = {
            "dishes": lambda: all(group in entry.dishes for group in query.dishes),
            "tastes": lambda: all(taste in entry.tastes for taste in query.tastes),
            "types": lambda: not query.types or ("sparkling" if entry.sparkling else "still") in query.types,
            "colors": lambda: not query.colors or entry.color in query.colors,
            "sugars": lambda: not query.sugars or entry.sugar in query.sugars,
            "regions": lambda: not query.regions or entry.region in query.regions,
            "grapes": lambda: not query.grapes or any(grape in entry.grapes for grape in query.grapes),
            "rating": lambda: query.min_rating is None or entry.rating >= query.min_rating,
        }
        return all(check() for name, check in checks.items() if name != skip)

    @staticmethod
    def _sort(items: list[Scored], sort: str) -> list[Scored]:
        if sort == "name":
            return sorted(items, key=lambda item: item.entry.name)
        # при равенстве — стабильный «перемешанный» порядок: у 674 вин рейтинг 5,0, и по алфавиту
        # наверху оказывалась одна линейка одного производителя
        if sort == "rating":
            return sorted(items, key=lambda item: (-item.entry.rating, item.entry.shuffle))
        # по релевантности; при равенстве — рейтинг, затем больше подтверждённых причин
        return sorted(items, key=lambda item: (-item.score, -item.entry.rating, -len(item.reasons), item.entry.shuffle))

    def _correct(self, text: str) -> str | None:
        """Исправить опечатки: каждое слово заменяем ближайшим словом каталога."""
        words = []
        for word in text.split():
            if any(known.startswith(word) for known in self._vocabulary):
                words.append(word)
                continue
            close = difflib.get_close_matches(word, self._vocabulary, n=1, cutoff=0.75)
            if not close:
                return None
            words.append(close[0])
        return " ".join(words)

    # --- счётчики вариантов --------------------------------------------------------------------

    def _facets(self, entries: list[Entry], query: SommelierQuery, text: str) -> SommelierFacets:
        def base(skip: str) -> list[Entry]:
            return [item.entry for item in self._run(entries, query, text, skip=skip)]

        def options(values: dict[str, str], counts: Counter, order: str = "given") -> list[SommelierOption]:
            items = [SommelierOption(value=value, label=label, count=counts.get(value, 0)) for value, label in values.items()]
            if order == "count":
                items.sort(key=lambda option: (-option.count, option.label))
            elif order == "alpha":
                items.sort(key=lambda option: option.label.lower().replace("ё", "е"))
            return items

        current = base(skip="__none__")
        occasion_base = base(skip="occasion")
        occasion_counts = Counter(
            key for entry in occasion_base for key in OCCASIONS if _occasion(entry, key)[0] >= MIN_OCCASION_SCORE
        )
        catalog_colors = sorted({entry.color for entry in entries if entry.color})
        catalog_sugars = sorted({entry.sugar for entry in entries if entry.sugar}, key=_sugar_order)
        catalog_regions = Counter(entry.region for entry in entries if entry.region)
        catalog_grapes = Counter(grape for entry in entries for grape in entry.grapes)
        return SommelierFacets(
            occasions=options({key: label for key, (label, _) in OCCASIONS.items()}, occasion_counts, order="alpha"),
            # блюда и вкусы складываются (И): счётчик — сколько останется, если добавить вариант
            dishes=options(
                {key: label for key, (label, _) in DISH_GROUPS.items()},
                Counter(group for entry in current for group in entry.dishes),
                order="alpha",
            ),
            tastes=options(
                {key: taste.label for key, taste in TASTES.items()},
                Counter(taste for entry in current for taste in entry.tastes),
                order="alpha",
            ),
            types=options(TYPES, Counter("sparkling" if e.sparkling else "still" for e in base("types"))),
            colors=options({c: c for c in catalog_colors}, Counter(e.color for e in base("colors"))),
            sugars=options({s: s for s in catalog_sugars}, Counter(e.sugar for e in base("sugars"))),
            regions=options(
                {region: region for region, _ in catalog_regions.most_common()},
                Counter(e.region for e in base("regions")),
            ),
            grapes=options(
                {grape: grape for grape, _ in catalog_grapes.most_common()},
                Counter(grape for e in base("grapes") for grape in e.grapes),
                order="count",
            ),
        )


def _occasion(entry: Entry, key: str) -> tuple[float, list[str]]:
    points, reasons = 0.0, []
    for condition, value, reason in OCCASIONS[key][1]:
        if condition(entry):
            points += value
            reasons.append(reason(entry))
    return points, reasons


def _text_score(entry: Entry, text: str, tokens: list[str]) -> float:
    """0 — не подходит; больше — ближе к названию."""
    latin_tokens = [to_latin(token) for token in tokens]
    if not all(t in entry.haystack or lt in entry.haystack_latin for t, lt in zip(tokens, latin_tokens)):
        return 0
    if entry.name == text:
        return 100
    if entry.name.startswith(text):
        return 60
    name_latin = to_latin(entry.name)
    if all(t in entry.name or lt in name_latin for t, lt in zip(tokens, latin_tokens)):
        return 40
    return 20  # совпало с производителем, сортом или регионом


def _sugar_order(sugar: str) -> int:
    order = ["экстра брют", "брют", "сухое", "полусухое", "полусладкое", "сладкое"]
    return order.index(sugar) if sugar in order else len(order)
