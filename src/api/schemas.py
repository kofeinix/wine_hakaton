from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class WineResponse(BaseModel):
    id: str
    name: str
    producer: str
    country: str
    region: str
    vintage: int | None = None
    grapes: list[str] = Field(default_factory=list)
    style: str | None = None
    average_rating: float = Field(ge=0, le=5)
    ratings_count: int = Field(ge=0)
    price: float | None = Field(default=None, ge=0)
    currency: str | None = None
    image_url: str | None = None
    description: str | None = None


class SearchMatch(BaseModel):
    wine: WineResponse
    score: float = Field(ge=0, le=1)
    reason: str


class SearchResponse(BaseModel):
    query_id: UUID
    filename: str | None = None
    content_type: str
    image_size_bytes: int = Field(ge=1)
    recognized_label: str | None = None
    results: list[SearchMatch]


class WineListResponse(BaseModel):
    items: list[WineResponse]
    total: int = Field(ge=0)


class ReviewCreate(BaseModel):
    author_name: str = Field(min_length=1, max_length=120)
    rating: float = Field(ge=0, le=5)
    text: str = Field(min_length=1, max_length=3000)


class ReviewResponse(BaseModel):
    id: str
    wine_id: str
    author_name: str
    rating: float = Field(ge=0, le=5)
    text: str
    created_at: datetime


class ReviewListResponse(BaseModel):
    items: list[ReviewResponse]
    total: int = Field(ge=0)


class DictionaryResponse(BaseModel):
    countries: list[str]
    grapes: list[str]
    styles: list[str]
