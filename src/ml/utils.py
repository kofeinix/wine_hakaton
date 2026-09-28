from io import BytesIO

import torch
from PIL import Image, ImageOps


def resolve_device(setting: str = "auto") -> str:
    """Устройство для моделей: явное из настроек или auto — cuda, затем mps (Apple), затем cpu."""
    if setting and setting != "auto":
        return setting
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def open_rgb_image(image_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def image_to_jpeg_bytes(image: Image.Image, max_side: int = 1024) -> bytes:
    image = image.copy()
    image.thumbnail((max_side, max_side))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    return buffer.getvalue()

