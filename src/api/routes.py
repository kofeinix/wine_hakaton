from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status

from src.api.schemas import (
    DictionaryResponse,
    ReviewCreate,
    ReviewListResponse,
    ReviewResponse,
    SearchResponse,
    WineListResponse,
    WineResponse,
)
from src.api.service import WineCatalogService

router = APIRouter()


def get_wine_service(request: Request) -> WineCatalogService:
    return request.app.state.wine_service


WineServiceDep = Annotated[WineCatalogService, Depends(get_wine_service)]


@router.post("/search/image", response_model=SearchResponse, tags=["search"])
async def search_by_image(
    service: WineServiceDep,
    image: UploadFile = File(..., description="Wine label image"),
    limit: int = Query(default=5, ge=1, le=20),
) -> SearchResponse:
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

    return service.search_by_image(
        image_bytes=image_bytes,
        filename=image.filename,
        content_type=image.content_type,
        limit=limit,
    )


@router.get("/wines", response_model=WineListResponse, tags=["wines"])
async def list_wines(
    service: WineServiceDep,
    q: str | None = Query(default=None, description="Search by wine name or producer"),
    country: str | None = None,
    grape: str | None = None,
    min_rating: float | None = Query(default=None, ge=0, le=5),
    limit: int = Query(default=20, ge=1, le=100),
) -> WineListResponse:
    return service.list_wines(
        q=q,
        country=country,
        grape=grape,
        min_rating=min_rating,
        limit=limit,
    )


@router.get("/wines/{wine_id}", response_model=WineResponse, tags=["wines"])
async def get_wine(
    wine_id: str,
    service: WineServiceDep,
) -> WineResponse:
    wine = service.get_wine(wine_id)
    if wine is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wine not found")
    return wine


@router.get(
    "/wines/{wine_id}/reviews",
    response_model=ReviewListResponse,
    tags=["reviews"],
)
async def get_wine_reviews(
    wine_id: str,
    service: WineServiceDep,
) -> ReviewListResponse:
    if service.get_wine(wine_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wine not found")
    return service.get_reviews(wine_id)


@router.post(
    "/wines/{wine_id}/reviews",
    response_model=ReviewResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["reviews"],
)
async def publish_review(
    wine_id: str,
    payload: ReviewCreate,
    service: WineServiceDep,
) -> ReviewResponse:
    if service.get_wine(wine_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wine not found")
    return service.create_review(wine_id=wine_id, payload=payload)


@router.get("/dictionaries", response_model=DictionaryResponse, tags=["metadata"])
async def get_dictionaries(
    service: WineServiceDep,
) -> DictionaryResponse:
    return service.get_dictionaries()
