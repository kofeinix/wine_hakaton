#!/usr/bin/env python3
"""Эксперименты с OCR-матчингом (кросс-валидация с группировкой по вину).

  export OCR_LAB_RESPONSES=data/eval_responses_eval.json   # по умолчанию старый дамп
  uv run python scripts/ocr_lab/experiments.py prepare     # признаки 7 матчеров + токены (~20 с)
  uv run python scripts/ocr_lab/experiments.py matchers    # CV всех алгоритмов сопоставления
  uv run python scripts/ocr_lab/experiments.py formula     # CV формул с "уникальными доказательствами"
  uv run python scripts/ocr_lab/experiments.py catboost    # CatBoost с нуля и поверх формулы
  uv run python scripts/ocr_lab/experiments.py errors      # разбор ошибок продовой формулы и потолок OCR
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def prepare() -> None:
    for script in ("lab_features.py", "lab_tokens.py"):
        subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=HERE)


def _load():
    import lab_eval as E

    return E, E.load()


def _matcher_job(args):
    E, D = _load()
    matcher, suffix, extra = args
    return args, E.cv(D, E.make_scorer(D, matcher, suffix, use_extra=extra))


def matchers() -> None:
    jobs = [(m, s, True) for m in ("win", "win1", "cov", "covt", "idf", "idfl", "part") for s in ("", "_notr")]
    jobs += [(m, "", False) for m in ("win", "covt", "idf")]
    with Pool(min(12, len(jobs))) as pool:
        results = pool.map(_matcher_job, jobs)
    for (m, s, extra), (acc, mrr) in sorted(results, key=lambda r: -r[1][0]):
        tr = "без транслит. OCR" if s else "транслит. OCR"
        print(f"{m:5} {tr:18} противоречия={'да' if extra else 'нет'}: CV Acc@1={acc:.2%} MRR={mrr:.4f}")


def _formula_job(args):
    import lab_spec as S

    E, D = _load()
    matcher, translit, seed = args
    sa = S.spec_arrays(D, translit)
    suffix = "" if translit else "_notr"
    return args, E.cv(D, S.make_scorer(D, sa, matcher, suffix), seeds=(seed,), space=S.SPACE)


def formula() -> None:
    E, D = _load()
    vis = D.X[:, D.col["visual"]]
    print("только изображение      ", "Acc@1={:.2%} MRR={:.4f}".format(*E.metrics(D, vis)))
    print("продовая формула (фикс.)", "Acc@1={:.2%} MRR={:.4f}".format(*E.metrics(D, vis + D.prod)))
    configs = [("idfl", True), ("idfl", False), ("cov", True), ("idf", True), ("win", True)]
    jobs = [(m, tr, seed) for m, tr in configs for seed in range(5)]
    with Pool(12) as pool:
        out = pool.map(_formula_job, jobs)
    for m, tr in configs:
        r = np.array([res for (mm, tt, _), res in out if (mm, tt) == (m, tr)])
        print(
            f"{m:5} + уник. доказательства, транслит={tr!s:5}: "
            f"CV Acc@1={r[:, 0].mean():.2%} ± {r[:, 0].std():.2%} MRR={r[:, 1].mean():.4f}"
        )


def catboost() -> None:
    from catboost import CatBoost
    from catboost import Pool as CPool

    import lab_spec as S

    E, D = _load()
    sa = S.spec_arrays(D, True)
    ncand = D.X.shape[0]
    spec = np.bincount(sa["cand"], weights=sa["idf"] * ((sa["s"] >= 0.78) & ~sa["isnum"]), minlength=ncand)
    stage1 = D.X[:, D.col["visual"]] + D.prod
    features = np.c_[D.X, D.prod, spec, stage1]
    print("формула этапа 1", "Acc@1={:.2%} MRR={:.4f}".format(*E.metrics(D, stage1)))
    for title, scale, depth, iters in (
        ("CatBoost с нуля", 0.0, 4, 400),
        ("CatBoost поверх формулы", 100.0, 4, 100),
    ):
        accs = []
        for seed in (0, 1):
            full = np.zeros(ncand)
            for train, test in E.folds(D, seed=seed):
                tr, te = train[D.rec_idx], test[D.rec_idx]
                model = CatBoost({
                    "loss_function": "YetiRank", "iterations": iters, "depth": depth, "learning_rate": 0.03,
                    "l2_leaf_reg": 10, "random_seed": 0, "verbose": False,
                })
                model.fit(CPool(features[tr], D.label[tr].astype(float), group_id=D.rec_idx[tr],
                                baseline=scale * stage1[tr] if scale else None))
                full[te] = model.predict(features[te]) + scale * stage1[te]
            accs.append(E.metrics(D, full))
        acc, mrr = np.mean(accs, axis=0)
        print(f"{title:24} CV Acc@1={acc:.2%} MRR={mrr:.4f}")


def errors() -> None:
    """Ошибки продовой формулы: причина и есть ли в OCR слово в пользу правильного вина."""
    from collections import Counter

    from lab_data import lab_cache

    E, D = _load()
    toks = pickle.loads((lab_cache() / "tokens.pkl").read_bytes())
    cat = E._CAT
    score = D.X[:, D.col["visual"]] + D.prod
    kinds: Counter = Counter()
    evidence: Counter = Counter()
    fixable = []
    for i, rec in enumerate(D.records):
        a, n = D.starts[i], len(rec["cands"])
        top, exp = int(np.argmax(score[a:a + n])), int(np.argmax(D.label[a:a + n]))
        if top == exp:
            continue
        we, wt = (cat["wines"][rec["cands"][j]["wine_id"]] for j in (exp, top))
        same_card = (we["name"], we["producer"], we["color"], we["sugar"]) == (
            wt["name"], wt["producer"], wt["color"], wt["sugar"])
        kinds["одинаковые карточки в БД" if same_card else
              "тот же производитель" if we["producer"] == wt["producer"] else "другой производитель"] += 1
        tokens = toks[i][True]
        found = {(f, k) for f, k, s in tokens[exp] if s >= 0.8}
        only_expected = found - {(f, k) for f, k, _ in tokens[top]}
        only_top = {(f, k) for f, k, s in tokens[top] if s >= 0.8} - {(f, k) for f, k, _ in tokens[exp]}
        if not only_expected:
            evidence["в OCR нет слов в пользу правильного (OCR не поможет)"] += 1
        elif only_top:
            evidence["слова в пользу обоих"] += 1
        else:
            evidence["в OCR есть слово только правильного (можно исправить)"] += 1
            fixable.append((rec["image"], we["name"], wt["name"], sorted(k for _, k in only_expected)))
    total = sum(kinds.values())
    print(f"ошибок: {total} из {D.n}")
    for key, value in kinds.most_common():
        print(f"  {value:3}  {key}")
    for key, value in evidence.most_common():
        print(f"  {value:3}  {key}")
    for image, expected, top, words in fixable:
        print(f"    {image}: '{expected}' проиграл '{top}', в OCR есть {words}")


COMMANDS = {"prepare": prepare, "matchers": matchers, "formula": formula, "catboost": catboost, "errors": errors}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
