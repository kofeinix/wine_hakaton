from io import BytesIO

from fastapi import UploadFile, HTTPException
from PIL import Image
from starlette import status


async def _read_image_upload(image: UploadFile) -> bytes:
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only image uploads are supported",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty",
        )
    # content-type задаёт клиент: битый файл иначе падает 500 глубоко в CV-пайплайне
    # (ultralytics подменяет Image.open и превращает ошибку декодирования в ModuleNotFoundError)
    try:
        with Image.open(BytesIO(image_bytes)) as decoded:
            decoded.verify()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Не удалось прочитать изображение",
        ) from exc
    return image_bytes