#!/usr/bin/env python3
"""Обучить адаптер векторов SigLIP2 по ракурсам → models/adapter/siglip2_384_views.npz.

Зачем. SigLIP2 обучен на общих картинках: фото бутылки «из жизни» (полка, руки, свет) и фото из каталога
одного и того же вина он считает менее похожими, чем хотелось бы, а соседние вина одной линейки — более.
Адаптер — небольшая поправка к вектору, f(x) = normalize(x + B·A·x) ранга 64, своя для каждого ракурса
(original, bottle_crop, label_crop). Сама модель SigLIP2 не меняется; адаптер применяется одинаково к фото
каталога (scripts/init_qdrant.py) и к запросу (src/ml/view_adapter.py).

Данные — готовые эмбеддинги каталога data/embeddings/siglip2_384/<view>.npz (scripts/index_siglip2_views_qdrant.py),
метка — только wine_id, ручной разметки нет:
- якорь — фото «как в жизни»: сгенерированные сцены (flux_*: полка / стол / в руках при разном свете),
  фото с Яндекса (yandex_*) и фото пользователей Vivino (vivino_*, отложенная выборка data/eval);
- позитив — любое другое фото того же вина (главное, другая сцена, другое фото);
- негативы — позитивы остальных вин батча; батч собирается из похожих вин (случайные вина + их ближайшие
  соседи по текущему индексу), так модель учится различать вина одной линейки.
Функция потерь — симметричный InfoNCE (температура 0.1). Не учим на шуме: фото, почти совпадающие
(косинус > 0.98) с фото другого вина, выбрасываются, а вина-дубли (средние векторы > 0.98) не считаются
друг другу негативами.

Проверка (--holdout, по умолчанию включена): сначала адаптер обучается без фото Vivino, и на них считается
top-1 вина по каждому ракурсу до и после адаптера (ближайшее фото каталога без Vivino). Потом — финальное
обучение на всех фото, включая Vivino; метрики проверки сохраняются в meta файла адаптера.
Полный поиск (ракурсы + OCR) на тех же 537 фото Vivino, адаптер без них: 88.8% → 92.9% top-1
(SigLIP2-224 без адаптера → SigLIP2-384 с адаптером).

    python scripts/train_view_adapter.py                # проверка + финальный адаптер
    python scripts/train_view_adapter.py --no-holdout   # только финальный адаптер

После обучения: docker compose up -d qdrant-init app (коллекции с новой версией адаптера загрузятся заново).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings" / "siglip2_384"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "adapter" / "siglip2_384_views.npz"
VIEWS = ("original", "bottle_crop", "label_crop")
HOLDOUT_SOURCE = "vivino"
DUPLICATE_COSINE = 0.98

logger = logging.getLogger(__name__)


def source_of(photo_id: str) -> str:
    """main, yandex, flux, vivino — по имени папки фото."""
    return photo_id.split("_", 1)[0]


def l2(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


class ViewData:
    def __init__(self, path: Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.vectors = l2(data["vectors"].astype("float32"))
        self.wine_ids = data["wine_ids"].astype(str)
        self.sources = np.array([source_of(p) for p in data["photo_ids"].astype(str)])
        self.wines, self.wine_index = np.unique(self.wine_ids, return_inverse=True)
        self.clean = ~self._cross_wine_duplicates()

    def _cross_wine_duplicates(self, chunk: int = 2048) -> np.ndarray:
        """Фото, почти совпадающее с фото другого вина: одна и та же картинка у разных вин — шум для обучения."""
        duplicate = np.zeros(len(self.vectors), dtype=bool)
        for start in range(0, len(self.vectors), chunk):
            sims = self.vectors[start : start + chunk] @ self.vectors.T
            other = self.wine_index[start : start + chunk, None] != self.wine_index[None, :]
            duplicate[start : start + chunk] = ((sims > DUPLICATE_COSINE) & other).any(axis=1)
        return duplicate


class Adapter(torch.nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.a = torch.nn.Linear(dim, rank, bias=False)
        self.b = torch.nn.Linear(rank, dim, bias=False)
        torch.nn.init.zeros_(self.b.weight)  # старт = исходные векторы SigLIP2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x + self.b(self.a(x)), dim=-1)


def apply(weights: tuple[np.ndarray, np.ndarray], vectors: np.ndarray) -> np.ndarray:
    a, b = weights
    return l2(vectors + (vectors @ a.T) @ b.T)


def train_view(data: ViewData, use: np.ndarray, args: argparse.Namespace, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """use — какие фото участвуют в обучении. Возвращает веса (A: r×D, B: D×r)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_wines = len(data.wines)
    photos = [[] for _ in range(n_wines)]
    for i in np.flatnonzero(use):
        photos[data.wine_index[i]].append(int(i))
    anchors = [[i for i in p if data.sources[i] != "main"] or p for p in photos]
    usable = np.array([len(p) >= 2 for p in photos])

    centroids = np.zeros((n_wines, data.vectors.shape[1]), dtype="float32")
    for w, p in enumerate(photos):
        if p:
            centroids[w] = data.vectors[p].mean(0)
    centroids = l2(centroids)
    sims = centroids @ centroids.T
    np.fill_diagonal(sims, -1)
    duplicate_wines = sims > DUPLICATE_COSINE
    neighbours = np.argsort(-np.where(duplicate_wines, -1, sims), axis=1)[:, : args.neighbours]

    x = torch.from_numpy(data.vectors)
    model = Adapter(x.shape[1], args.rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    target = torch.arange(args.batch)
    for step in range(1, args.steps + 1):
        batch: list[int] = []
        seen: set[int] = set()
        while len(batch) < args.batch:
            seed_wine = int(rng.integers(n_wines))
            for w in (seed_wine, *neighbours[seed_wine][: args.per_seed - 1]):
                w = int(w)
                if usable[w] and w not in seen and len(batch) < args.batch:
                    seen.add(w)
                    batch.append(w)
        anchor_idx, positive_idx = [], []
        for w in batch:
            a = int(rng.choice(anchors[w]))
            anchor_idx.append(a)
            positive_idx.append(int(rng.choice([i for i in photos[w] if i != a])))
        logits = model(x[anchor_idx]) @ model(x[positive_idx]).T / args.temperature
        logits = logits.masked_fill(torch.from_numpy(duplicate_wines[np.ix_(batch, batch)]), -1e4)
        loss = (torch.nn.functional.cross_entropy(logits, target) + torch.nn.functional.cross_entropy(logits.T, target)) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == args.steps:
            logger.info("  шаг %s/%s, loss %.3f", step, args.steps, loss.item())
    return (
        model.a.weight.detach().numpy().astype("float32"),
        model.b.weight.detach().numpy().astype("float32"),
    )


def top1_wine(queries: np.ndarray, catalog: np.ndarray, catalog_wines: np.ndarray) -> np.ndarray:
    """Вино ближайшего фото каталога (= top-1 вина по максимуму сходства)."""
    return catalog_wines[np.argmax(queries @ catalog.T, axis=1)]


def holdout_check(data: dict[str, ViewData], args: argparse.Namespace) -> dict:
    """Адаптер без фото Vivino; top-1 на них по каждому ракурсу до и после."""
    report = {}
    for view, d in data.items():
        holdout = d.sources == HOLDOUT_SOURCE
        train = ~holdout & d.clean
        queries = holdout & d.clean
        logger.info("%s: проверка — обучение на %s фото, отложено %s фото Vivino", view, train.sum(), queries.sum())
        weights = train_view(d, train, args, args.seed)
        catalog = ~holdout
        truth = d.wine_ids[queries]
        before = top1_wine(d.vectors[queries], d.vectors[catalog], d.wine_ids[catalog]) == truth
        adapted = apply(weights, d.vectors)
        after = top1_wine(adapted[queries], adapted[catalog], d.wine_ids[catalog]) == truth
        report[view] = {
            "queries": int(queries.sum()),
            "top1_before": round(float(before.mean()) * 100, 1),
            "top1_after": round(float(after.mean()) * 100, 1),
            "fixed": int((~before & after).sum()),
            "broke": int((before & ~after).sum()),
        }
        logger.info("%s: top-1 по фото Vivino %.1f%% → %.1f%%", view, report[view]["top1_before"], report[view]["top1_after"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-holdout", dest="holdout", action="store_false", help="Без проверки на отложенных фото Vivino")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=256, help="Вин в батче")
    parser.add_argument("--per-seed", type=int, default=8, help="Вин в группе похожих: случайное + ближайшие соседи")
    parser.add_argument("--neighbours", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    started = time.perf_counter()
    data = {}
    for view in VIEWS:
        d = ViewData(args.embeddings_dir / f"{view}.npz")
        counts = {str(s): int((d.sources == s).sum()) for s in sorted(set(d.sources))}
        logger.info(
            "%s: %s фото %s, вин %s, отброшено дублей между винами %s",
            view, len(d.vectors), counts, len(d.wines), int((~d.clean).sum()),
        )
        data[view] = d

    holdout = holdout_check(data, args) if args.holdout else None

    arrays: dict[str, np.ndarray] = {}
    trained_on = {}
    for view, d in data.items():
        logger.info("%s: финальное обучение на %s фото (вместе с Vivino)", view, int(d.clean.sum()))
        a, b = train_view(d, d.clean, args, args.seed)
        arrays[f"{view}/a"], arrays[f"{view}/b"] = a, b
        trained_on[view] = {str(s): int(((d.sources == s) & d.clean).sum()) for s in sorted(set(d.sources))}

    meta = {
        "model": "google/siglip2-base-patch16-384",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "formula": "normalize(x + B @ (A @ x))",
        "params": {k: v for k, v in vars(args).items() if k not in ("embeddings_dir", "output", "holdout")},
        "trained_on": trained_on,
        "holdout_vivino": holdout,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    tmp.replace(args.output)
    logger.info("Адаптер сохранён: %s (%.0f КБ), %.0f с", args.output, args.output.stat().st_size / 1024, time.perf_counter() - started)


if __name__ == "__main__":
    main()
