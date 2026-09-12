from io import BytesIO
from PIL import Image, ImageOps


def open_rgb_image(image_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def image_to_jpeg_bytes(image: Image.Image, max_side: int = 1024) -> bytes:
    image = image.copy()
    image.thumbnail((max_side, max_side))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    return buffer.getvalue()

