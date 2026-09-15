from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    CompactSearchResponse,
    SearchResponse,
    WineResponse,
)
from src.api.services import WineService
from src.api.utils import _read_image_upload

router = APIRouter()


def get_wine_service(request: Request) -> WineService:
    return request.app.state.text_service


WineServiceDep = Annotated[WineService, Depends(get_wine_service)]


@router.post(
    "/search/image",
    response_model=CompactSearchResponse,
    tags=["search"],
    summary="Search wine by image",
    description=(
        "Accepts a wine bottle or label image and returns compact matches as "
        "`result: [{wine_id, score}]`. Use `/search/image/extended` for crop, extraction, "
        "source, and full wine details."
    ),
    responses={
        400: {"description": "Uploaded image is empty"},
        415: {"description": "Uploaded file is not an image"},
    },
)
async def search_by_image(
    wine_service: WineServiceDep,
    image: UploadFile = File(..., description="Wine bottle or label image"),
    limit: int = Query(default=5, ge=1, le=20, description="Maximum number of matches"),
) -> CompactSearchResponse:
    image_bytes = await _read_image_upload(image)
    return await wine_service.search_by_image(
        image_bytes=image_bytes,
        limit=limit,
    )


@router.post(
    "/search/image/extended",
    response_model=SearchResponse,
    tags=["search"],
    summary="Search wine by image with diagnostics",
    description=(
        "Runs the same image search pipeline as `/search/image`, but returns YOLO crop metadata, "
        "NuExtract fields from the full image and crop, full wine records, and source scores."
    ),
    responses={
        400: {"description": "Uploaded image is empty"},
        415: {"description": "Uploaded file is not an image"},
    },
)
async def search_by_image_extended(
    wine_service: WineServiceDep,
    image: UploadFile = File(..., description="Wine bottle or label image"),
    limit: int = Query(default=5, ge=1, le=20, description="Maximum number of matches"),
) -> SearchResponse:
    image_bytes = await _read_image_upload(image)
    return await wine_service.search_by_image_extended(
        image_bytes=image_bytes,
        filename=image.filename,
        content_type=image.content_type or "application/octet-stream",
        limit=limit,
    )


@router.get(
    "/wines/{wine_id}",
    response_model=WineResponse,
    tags=["wines"],
    summary="Get wine by id",
    responses={404: {"description": "Wine not found"}},
)
async def get_wine(
    wine_id: str,
    wine_service: WineServiceDep,
) -> WineResponse:
    wine = await wine_service.get_wine(wine_id)
    if wine is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wine not found")
    return wine


@router.get(
    "/wines/{wine_id}/photos/{filename}",
    tags=["wines"],
    summary="Get wine photo",
    responses={404: {"description": "Wine photo not found"}},
)
async def get_wine_photo(
    wine_id: str,
    filename: str,
    wine_service: WineServiceDep,
) -> StreamingResponse:
    photo = await wine_service.get_photo_file_by_wine_id(wine_id, filename)
    if photo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wine photo not found")

    file_data, content_type = photo
    return StreamingResponse(file_data, media_type=content_type)
