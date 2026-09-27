"""Каталог достижений — в коде: новое достижение = новая запись в build_catalog().

Прогресс не накапливается по событиям, а считается по данным пользователя (история сканов,
отзывы, реакции, избранное) — см. Snapshot. Поэтому он всегда совпадает с фактами и не требует
пересчёта задним числом.

Видимость (скрытые достижения не должны подсказывать, что делать):
- полученное видно всегда;
- в серии (1 → 10 → 100 …) из неполученных видно только следующее: если оно не скрытое или
  предыдущее уже получено (пользователь «вошёл» в серию — например, отсканировал первое вино Кубани);
- одиночное скрытое видно только после получения.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from src.api.services.sommelier_service import Entry, to_latin

# скан засчитывается, если фото снято кнопкой «Камера» только что (флаг from_camera в истории)
# и уверенность не ниже 70%
SCAN_MIN_CONFIDENCE = 0.7
HIGH_CONFIDENCE = 0.9
REGION_TIERS = (1, 10, 100, 200, 500)
RARE_GRAPE_MAX_WINES = 3

CATEGORIES = {
    "scans": "Сканирование",
    "regions": "Регионы",
    "styles": "Цвет и стиль",
    "grapes": "Сорта винограда",
    "producers": "Винодельни",
    "social": "Отзывы и реакции",
    "favorites": "Избранное",
    "time": "Время и сезоны",
    "meta": "Коллекция",
}


@dataclass
class Scan:
    wine: Entry
    at: datetime  # в часовом поясе пользователя
    confidence: float


@dataclass
class Snapshot:
    """Всё, из чего считается прогресс, для одного пользователя."""

    scans: list[Scan] = field(default_factory=list)  # засчитанные сканы (уверенность ≥ 70%)
    comments: int = 0
    ratings: set[int] = field(default_factory=set)
    review_from_notification: bool = False
    likes_given: int = 0
    dislikes_given: int = 0
    max_likes_on_comment: int = 0
    favorites: int = 0
    favorite_after_scan: bool = False
    earned: set[str] = field(default_factory=set)  # заполняется перед подсчётом мета-достижений

    # --- производные -------------------------------------------------------------------

    @property
    def wines(self) -> list[Entry]:
        unique = {}
        for scan in self.scans:
            unique.setdefault(scan.wine.card.id, scan.wine)
        return list(unique.values())

    def count(self, predicate: Callable[[Entry], bool]) -> int:
        return sum(1 for wine in self.wines if predicate(wine))

    def distinct(self, key: Callable[[Entry], Iterable[str | None]]) -> int:
        return len({value for wine in self.wines for value in key(wine) if value})

    def longest_streak(self) -> int:
        days = sorted({scan.at.date() for scan in self.scans})
        best = run = 0
        previous: date | None = None
        for day in days:
            run = run + 1 if previous and day - previous == timedelta(days=1) else 1
            best = max(best, run)
            previous = day
        return best

    def most_wines_in_a_day(self) -> int:
        per_day: dict[date, set[str]] = {}
        for scan in self.scans:
            per_day.setdefault(scan.at.date(), set()).add(scan.wine.card.id)
        return max((len(ids) for ids in per_day.values()), default=0)

    def any_scan(self, predicate: Callable[[Scan], bool]) -> int:
        return int(any(predicate(scan) for scan in self.scans))


@dataclass(frozen=True)
class Achievement:
    code: str
    title: str
    description: str
    category: str
    target: int
    progress: Callable[[Snapshot], int]
    hidden: bool = False
    series: str | None = None
    meta: bool = False  # считается после остальных (зависит от числа полученных)


def build_catalog(entries: list[Entry]) -> list[Achievement]:
    """Реестр достижений; цели «все вина региона», «все цвета» и т. п. берутся из каталога вин."""
    items: list[Achievement] = []

    def add(code, title, description, category, target, progress, *, hidden=False, series=None, meta=False):
        items.append(Achievement(code, title, description, category, target, progress, hidden, series, meta))

    def wines(predicate: Callable[[Entry], bool]) -> Callable[[Snapshot], int]:
        return lambda snap: snap.count(predicate)

    def has_grape(fragment: str) -> Callable[[Entry], bool]:
        return lambda wine: any(fragment in grape for grape in wine.grapes)

    unique_wines = wines(lambda wine: True)

    # --- сканирование -------------------------------------------------------------------
    for target, code, title, hidden in (
        (1, "first_scan", "Первый шаг", True),
        (5, "scan_5_wines", "Вошёл во вкус", False),
        (25, "scan_25_wines", "Домашний сомелье", False),
        (100, "scan_100_wines", "Винный исследователь", False),
        (500, "scan_500_wines", "Архивариус этикеток", True),
        (1000, "scan_1000_wines", "Живая энциклопедия", True),
    ):
        description = "Найти вино по фото" if target == 1 else f"Найти по фото {target} разных вин"
        add(code, title, description, "scans", target, unique_wines, hidden=hidden, series="scans")
    add(
        "high_confidence_scan",
        "Чёткий кадр",
        "Скан с уверенностью от 90%",
        "scans",
        1,
        lambda snap: snap.any_scan(lambda scan: scan.confidence >= HIGH_CONFIDENCE),
        series="high_confidence",
    )
    add(
        "high_confidence_10",
        "Острый фокус",
        "10 сканов с уверенностью от 90%",
        "scans",
        10,
        lambda snap: sum(1 for scan in snap.scans if scan.confidence >= HIGH_CONFIDENCE),
        hidden=True,
        series="high_confidence",
    )

    # --- регионы: серия 1 / 10 / 100 / 200 / 500 / все вина региона -------------------------
    region_sizes = Counter(entry.region for entry in entries if entry.region)
    for region, size in region_sizes.most_common():
        slug = "_".join(to_latin(region.lower()).replace("—", " ").split())
        tiers = sorted({tier for tier in REGION_TIERS if tier < size} | {size})
        for tier in tiers:
            if tier == size:
                title, description = f"Все вина: {region}", f"Найти все {size} вин региона «{region}» из каталога"
            elif tier == 1:
                title, description = f"{region}: первое вино", f"Найти вино региона «{region}»"
            else:
                title, description = f"{region}: {tier} вин", f"Найти {tier} разных вин региона «{region}»"
            add(
                f"region_{slug}_{'all' if tier == size else tier}",
                title,
                description,
                "regions",
                tier,
                wines(lambda wine, region=region: wine.region == region),
                hidden=True,  # регион открывается первым найденным вином
                series=f"region_{slug}",
            )
    all_regions = len(region_sizes)
    regions_found = lambda snap: snap.distinct(lambda wine: [wine.region])  # noqa: E731
    add("regions_3", "Три региона", "Вина из 3 разных регионов", "regions", 3, regions_found, series="regions")
    add("regions_7", "Винная карта", "Вина из 7 разных регионов", "regions", 7, regions_found, series="regions")
    add(
        "regions_all",
        "Географ вина",
        f"Вина из всех {all_regions} регионов каталога",
        "regions",
        all_regions,
        regions_found,
        hidden=True,
        series="regions",
    )

    # --- цвет, стиль, сахар -------------------------------------------------------------------
    colors = Counter(entry.color for entry in entries if entry.color)
    for color, code, name, tiers in (
        ("Красное", "red", "Красн", ((10, "Красная линия", False), (100, "Красная коллекция", True), (None, "Красная книга", True))),
        ("Белое", "white", "Бел", ((10, "Белая глава", False), (100, "Белая коллекция", True))),
        ("Розовое", "rose", "Розов", ((10, "Розовый период", False),)),
        ("Оранжевое", "orange", "Оранжев", ((5, "Оранжевый поворот", True),)),
    ):
        for target, title, hidden in tiers:
            target = target or colors[color]  # None — все вина этого цвета в каталоге
            add(
                f"{code}_wines_{target if target != colors[color] else 'all'}",
                title,
                f"Найти {target} разных {name.lower()}ых вин" + (" — все в каталоге" if target == colors[color] else ""),
                "styles",
                target,
                wines(lambda wine, color=color: wine.color == color),
                hidden=hidden,
                series=f"{code}_wines",
            )
    add("sparkling_wines_10", "Игристое настроение", "Найти 10 разных игристых вин", "styles", 10, wines(lambda w: w.sparkling))
    add("dry_wines_25", "Сухой закон наоборот", "Найти 25 разных сухих вин", "styles", 25, wines(lambda w: w.sugar == "сухое"))
    add(
        "sweet_wines_10",
        "Сладкая полка",
        "Найти 10 сладких или полусладких вин",
        "styles",
        10,
        wines(lambda w: w.sugar in {"сладкое", "полусладкое"}),
    )
    add(
        "all_colors",
        "Вся палитра",
        f"Найти вино каждого цвета: {', '.join(color.lower() for color in sorted(colors))}",
        "styles",
        len(colors),
        lambda snap: snap.distinct(lambda wine: [wine.color]),
        hidden=True,
    )
    sugars = {entry.sugar for entry in entries if entry.sugar}
    add(
        "all_sugar_styles",
        "Баланс вкуса",
        "Найти вина всех категорий по сахару: от экстра брют до сладкого",
        "styles",
        len(sugars),
        lambda snap: snap.distinct(lambda wine: [wine.sugar]),
        hidden=True,
    )

    # --- сорта ----------------------------------------------------------------------------------
    for code, title, fragment, hidden in (
        ("saperavi_10", "Саперави в сердце", "Саперави", False),
        ("krasnostop_10", "Красностоп найден", "Красностоп", True),
        ("riesling_10", "Рислинг-радар", "Рислинг", False),
        ("chardonnay_10", "Шардоне-клуб", "Шардоне", False),
        ("pinot_noir_10", "Пино-охотник", "Пино Нуар", True),
    ):
        add(code, title, f"Найти 10 разных вин с сортом {fragment}", "grapes", 10, wines(has_grape(fragment)), hidden=hidden)
    grapes_found = lambda snap: snap.distinct(lambda wine: wine.grapes)  # noqa: E731
    add("grapes_10", "Ампелограф-новичок", "Вина с 10 разными сортами винограда", "grapes", 10, grapes_found, series="grapes")
    add("grapes_50", "Ампелограф", "Вина с 50 разными сортами винограда", "grapes", 50, grapes_found, hidden=True, series="grapes")
    grape_sizes = Counter(grape for entry in entries for grape in entry.grapes)
    rare = {grape for grape, count in grape_sizes.items() if count <= RARE_GRAPE_MAX_WINES}
    add(
        "rare_grape",
        "Редкий сорт",
        f"Найти вино с сортом, который встречается в каталоге не больше {RARE_GRAPE_MAX_WINES} раз",
        "grapes",
        1,
        lambda snap: int(any(grape in rare for wine in snap.wines for grape in wine.grapes)),
        hidden=True,
    )

    # --- винодельни -----------------------------------------------------------------------------
    producers_found = lambda snap: snap.distinct(lambda wine: [wine.card.producer])  # noqa: E731
    for target, code, title, hidden in (
        (5, "producers_5", "Пять виноделен", False),
        (25, "producers_25", "Винодельческий тур", False),
        (100, "producers_100", "Большой тур", True),
    ):
        add(code, title, f"Вина {target} разных производителей", "producers", target, producers_found, hidden=hidden, series="producers")
    add(
        "same_producer_10",
        "Верный выбор",
        "10 разных вин одного производителя",
        "producers",
        10,
        lambda snap: max(Counter(wine.card.producer for wine in snap.wines).values(), default=0),
        hidden=True,
    )

    # --- отзывы и реакции -----------------------------------------------------------------------
    for target, code, title, hidden in (
        (1, "first_review", "Первое мнение", False),
        (10, "reviews_10", "Голос дегустатора", False),
        (50, "reviews_50", "Критик сервиса", True),
        (100, "reviews_100", "Авторитетный голос", True),
    ):
        description = "Оставить первый комментарий к вину" if target == 1 else f"Оставить {target} комментариев к винам"
        add(code, title, description, "social", target, lambda snap: snap.comments, hidden=hidden, series="reviews")
    add("five_star_review", "Восторг в бокале", "Поставить вину 5 бокалов", "social", 1, lambda snap: int(5 in snap.ratings))
    add(
        "balanced_critic",
        "Честный критик",
        "Поставить оценки от 1 до 5 — каждую хотя бы раз",
        "social",
        5,
        lambda snap: len(snap.ratings),
        hidden=True,
    )
    for target, code, title, hidden in (
        (1, "comment_likes_1", "Вас услышали", False),
        (10, "comment_likes_10", "Вас заметили", True),
        (50, "comment_likes_50", "Мнение лидера", True),
    ):
        description = "Ваш комментарий получил лайк" if target == 1 else f"Один ваш комментарий получил {target} лайков"
        add(code, title, description, "social", target, lambda snap: snap.max_likes_on_comment, hidden=hidden, series="comment_likes")
    add("liked_others_10", "Щедрый лайк", "Поставить 10 лайков чужим комментариям", "social", 10, lambda snap: snap.likes_given)
    add(
        "disliked_others_10",
        "Строгий судья",
        "Поставить 10 дизлайков чужим комментариям",
        "social",
        10,
        lambda snap: snap.dislikes_given,
        hidden=True,
    )
    add(
        "civil_balance",
        "Баланс мнений",
        "Поставить чужим комментариям 5 лайков и 5 дизлайков",
        "social",
        10,
        lambda snap: min(snap.likes_given, 5) + min(snap.dislikes_given, 5),
        hidden=True,
    )
    add(
        "review_after_notification",
        "Вернулся с мнением",
        "Оценить вино из напоминания «вы недавно смотрели»",
        "social",
        1,
        lambda snap: int(snap.review_from_notification),
        hidden=True,
    )

    # --- избранное ------------------------------------------------------------------------------
    for target, code, title, hidden in (
        (1, "first_favorite", "На полку", False),
        (10, "favorites_10", "Личная полка", False),
        (50, "favorites_50", "Домашний погреб", True),
    ):
        description = "Добавить вино в избранное" if target == 1 else f"Добавить {target} вин в избранное"
        add(code, title, description, "favorites", target, lambda snap: snap.favorites, hidden=hidden, series="favorites")
    add(
        "favorite_after_scan",
        "Нашёл и сохранил",
        "Добавить в избранное вино из результата скана",
        "favorites",
        1,
        lambda snap: int(snap.favorite_after_scan),
    )

    # --- время и сезоны (по часовому поясу пользователя) ----------------------------------------
    streak = lambda snap: snap.longest_streak()  # noqa: E731
    add("scan_streak_3", "Три дня с вином", "Сканировать вино 3 дня подряд", "time", 3, streak, series="streak")
    add("scan_streak_7", "Недельная дегустация", "Сканировать вино 7 дней подряд", "time", 7, streak, hidden=True, series="streak")
    add("scan_streak_30", "Месяц открытий", "Сканировать вино 30 дней подряд", "time", 30, streak, hidden=True, series="streak")
    add(
        "scan_late_night",
        "Ночной дегустатор",
        "Скан между полуночью и пятью утра",
        "time",
        1,
        lambda snap: snap.any_scan(lambda scan: scan.at.hour < 5),
        hidden=True,
    )
    add(
        "scan_morning",
        "Утренний выбор",
        "Скан между 6 и 10 утра",
        "time",
        1,
        lambda snap: snap.any_scan(lambda scan: 6 <= scan.at.hour < 10),
        hidden=True,
    )
    add(
        "scan_many_one_day",
        "Винный марафон",
        "Найти 20 разных вин за один день",
        "time",
        20,
        lambda snap: snap.most_wines_in_a_day(),
        hidden=True,
    )
    add(
        "weekend_scan",
        "Выходной выбор",
        "Скан в субботу или воскресенье",
        "time",
        1,
        lambda snap: snap.any_scan(lambda scan: scan.at.weekday() >= 5),
    )
    add(
        "new_year_sparkling",
        "Праздничный звон",
        "Найти игристое в декабре или январе",
        "time",
        1,
        lambda snap: snap.any_scan(lambda scan: scan.at.month in (12, 1) and scan.wine.sparkling),
        hidden=True,
    )
    add(
        "summer_white",
        "Летняя свежесть",
        "Найти белое вино летом",
        "time",
        1,
        lambda snap: snap.any_scan(lambda scan: scan.at.month in (6, 7, 8) and scan.wine.color == "Белое"),
        hidden=True,
    )
    add(
        "autumn_red",
        "Осенний красный",
        "Найти красное вино осенью",
        "time",
        1,
        lambda snap: snap.any_scan(lambda scan: scan.at.month in (9, 10, 11) and scan.wine.color == "Красное"),
        hidden=True,
    )

    # --- коллекция: считаются после остальных --------------------------------------------------
    regular = [item for item in items]
    hidden_codes = {item.code for item in regular if item.hidden}
    earned_regular = lambda snap: sum(1 for item in regular if item.code in snap.earned)  # noqa: E731
    for target, code, title, hidden in (
        (5, "achievements_5", "Первые награды", False),
        (25, "achievements_25", "Коллекционер", False),
        (50, "achievements_50", "Охотник за достижениями", True),
    ):
        add(code, title, f"Получить {target} достижений", "meta", target, earned_regular, hidden=hidden, series="achievements", meta=True)
    add(
        "achievements_all",
        "Закрыть карту",
        "Получить все достижения",
        "meta",
        len(regular),
        earned_regular,
        hidden=True,
        series="achievements",
        meta=True,
    )
    add(
        "secret_achievement",
        "Тайная пробка",
        "Получить секретное достижение",
        "meta",
        1,
        lambda snap: int(bool(hidden_codes & snap.earned)),
        hidden=True,
        meta=True,
    )
    return items
