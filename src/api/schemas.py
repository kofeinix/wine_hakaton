from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WinePhotoResponse(BaseModel):
    id: str
    url: str
    object_name: str
    filename: str
    is_main: bool
    source_url: str | None = None


class WineResponse(BaseModel):
    id: str
    slug: str
    sku: str
    name: str
    producer: str | None = None
    region: str | None = None
    country: str | None = None
    vintage: int | None = None
    year: int | None = None
    color: str | None = None
    sugar: str | None = None
    style: str | None = None
    wine_type: str | None = None
    alcohol: float | None = None
    price: float | None = None
    currency: str = "RUB"
    stock: int | None = None
    rating: float | None = None
    description: str | None = None
    url: str | None = None
    source_url: str | None = None
    image_url: str | None = None
    grapes: list[str] = Field(default_factory=list)
    photos: list[WinePhotoResponse] = Field(default_factory=list)


class SearchMatchResponse(BaseModel):
    wine_id: str
    slug: str
    score: float
    max_score: float
    mean_score: float
    n_photos: int
    score_std: float
    cosine_score: float = 0.0
    wine: WineResponse | None = None


class CompactSearchMatch(BaseModel):
    wine_id: str
    slug: str
    score: float
    max_score: float
    mean_score: float
    n_photos: int
    score_std: float
    cosine_score: float = 0.0


class CompactSearchResponse(BaseModel):
    result: list[CompactSearchMatch]


class CropResponse(BaseModel):
    available: bool
    box: tuple[int, int, int, int] | None = None
    confidence: float | None = None
    width: int | None = None
    height: int | None = None


class SearchResponse(BaseModel):
    status: str
    slug: str | None = None
    confidence: float = 0.0
    gap: float = 0.0
    wine: WineResponse | None = None
    top5: list[dict[str, float | str]] = Field(default_factory=list)
    results: list[SearchMatchResponse] = Field(default_factory=list)
    crops: dict[str, CropResponse] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
