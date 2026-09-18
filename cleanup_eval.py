#!/usr/bin/env python3
"""Очистка data/eval:
1. Удаляет все папки from_yandex (рекурсивно).
2. Переносит файлы из convert_jpg и from_vivino на уровень выше (в папку uuid).
3. Удаляет файлы 0.webp.
4. Удаляет папки convert_jpg и from_vivino после успешного переноса.
5. Проверяет, что все файлы действительно перенесены.
"""
import shutil
from pathlib import Path

ROOT = Path("data/eval")

moved = 0
deleted_yandex = 0
deleted_webp = 0
deleted_dirs = 0
errors = []

for uuid_dir in sorted(ROOT.iterdir()):
    if not uuid_dir.is_dir():
        continue

    # 1. Удаляем from_yandex
    yandex_dir = uuid_dir / "from_yandex"
    if yandex_dir.is_dir():
        shutil.rmtree(yandex_dir)
        deleted_yandex += 1

    # 2. Переносим файлы из convert_jpg и from_vivino на уровень выше
    for sub in ("convert_jpg", "from_vivino"):
        sub_dir = uuid_dir / sub
        if not sub_dir.is_dir():
            continue
        for f in sorted(sub_dir.iterdir()):
            if not f.is_file():
                continue
            dest = uuid_dir / f.name
            if dest.exists():
                errors.append(f"Конфликт имени: {dest} уже существует (источник {f})")
                continue
            shutil.move(str(f), str(dest))
            moved += 1

    # 3. Удаляем 0.webp
    webp = uuid_dir / "0.webp"
    if webp.is_file():
        webp.unlink()
        deleted_webp += 1

    # 4. Удаляем пустые папки convert_jpg и from_vivino
    for sub in ("convert_jpg", "from_vivino"):
        sub_dir = uuid_dir / sub
        if sub_dir.is_dir():
            # Проверяем, что внутри ничего не осталось
            remaining = [p for p in sub_dir.iterdir()]
            if remaining:
                errors.append(f"Папка {sub_dir} не пуста, не удаляю: {remaining}")
                continue
            sub_dir.rmdir()
            deleted_dirs += 1

print(f"Перенесено файлов: {moved}")
print(f"Удалено папок from_yandex: {deleted_yandex}")
print(f"Удалено файлов 0.webp: {deleted_webp}")
print(f"Удалено пустых папок convert_jpg/from_vivino: {deleted_dirs}")

if errors:
    print("\nОШИБКИ:")
    for e in errors:
        print(" -", e)
else:
    print("\nВсе операции выполнены успешно, ошибок нет.")