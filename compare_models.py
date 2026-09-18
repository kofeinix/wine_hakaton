"""Сравнение моделей label.onnx и yolo26l-label.onnx на data/eval.

Рекурсивно обходит все изображения в data/eval, для каждой модели считает:
- в скольких случаях найден bbox (детекция этикетки)
- среднее время inference
- общее время inference
"""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.ml.yolo import YoloLabelCropper

_best_detection = YoloLabelCropper._best_detection
_preprocess = YoloLabelCropper._preprocess

EVAL_DIR = Path("data/eval")
MODELS = {
    "label.onnx": Path("models/yolo/label.onnx"),
    "yolo26l-label.onnx": Path("models/yolo/yolo26l-label.onnx"),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(root: Path) -> list[Path]:
    images = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
    return images


def load_session(model_path: Path) -> tuple[ort.InferenceSession, str, tuple[int, int]]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    shape = model_input.shape
    height = int(shape[2]) if isinstance(shape[2], int) else 640
    width = int(shape[3]) if isinstance(shape[3], int) else 640
    input_size = (width, height)
    return session, input_name, input_size


def run_model(session, input_name, input_size, image: Image.Image):
    """Возвращает (найден_bbox, время_inference_сек)."""
    input_tensor, _ = _preprocess(image, input_size)
    start = time.perf_counter()
    outputs = session.run(None, {input_name: input_tensor})
    elapsed = time.perf_counter() - start
    prediction = np.asarray(outputs[0]).squeeze()
    detection = _best_detection(prediction)
    return detection is not None, elapsed


def main() -> None:
    images = collect_images(EVAL_DIR)
    print(f"Найдено изображений в {EVAL_DIR}: {len(images)}")
    print()

    results = {}
    for name, model_path in MODELS.items():
        print(f"=== {name} ===")
        session, input_name, input_size = load_session(model_path)
        print(f"  input: {input_name} {input_size}")
        for i, out in enumerate(session.get_outputs()):
            print(f"  output[{i}]: {out.name} shape={out.shape}")
        print()

        found = 0
        total_time = 0.0
        times = []
        for img_path in images:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                is_found, elapsed = run_model(session, input_name, input_size, img)
            times.append(elapsed)
            total_time += elapsed
            if is_found:
                found += 1

        avg_time = total_time / len(times) if times else 0.0
        results[name] = {
            "found": found,
            "total": len(images),
            "avg_time": avg_time,
            "total_time": total_time,
        }
        print(
            f"  Найдено bbox: {found}/{len(images)} "
            f"({100.0 * found / len(images):.1f}%)"
        )
        print(f"  Среднее время inference: {avg_time * 1000:.2f} ms")
        print(f"  Общее время inference: {total_time:.2f} s")
        print()

    print("=== СРАВНЕНИЕ ===")
    names = list(results.keys())
    a, b = names[0], names[1]
    ra, rb = results[a], results[b]
    print(f"{'Метрика':<28}{a:<22}{b:<22}")
    print("-" * 72)
    print(f"{'Найдено bbox':<28}{ra['found']:<22}{rb['found']:<22}")
    print(
        f"{'Доля найденных':<28}"
        f"{100.0 * ra['found'] / ra['total']:.1f}%{'':<18}"
        f"{100.0 * rb['found'] / rb['total']:.1f}%"
    )
    print(
        f"{'Среднее время, ms':<28}"
        f"{ra['avg_time'] * 1000:<22.2f}{rb['avg_time'] * 1000:<22.2f}"
    )
    print(
        f"{'Общее время, s':<28}"
        f"{ra['total_time']:<22.2f}{rb['total_time']:<22.2f}"
    )


if __name__ == "__main__":
    main()