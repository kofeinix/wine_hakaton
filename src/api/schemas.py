from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class WinePhotoResponse(BaseModel):
    object_name: str
    filename: str
    url: str
    is_main: bool = False
    size_bytes: int | None = None
    content_type: str | None = None


class WinePhotoListResponse(BaseModel):
    wine_id: str
    photos: list[WinePhotoResponse] = Field(default_factory=list)


class WineResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "664766e5-a73d-5609-a0d7-cffbbe62d466",
                "name": "Cantiani Cabernet Sauvignon",
                "producer": "Виноградники Гай-Кодзора",
                "region": "Кубань",
                "grapes": ["Каберне Совиньон"],
                "style": "Красное сухое",
                "average_rating": 4.75,
                "ratings_count": 0,
                "description": "Округлый, хорошо сбалансированный вкус.",
                "wine_type": "Красное сухое",
                "alcohol": "12%",
                "image_url": "/api/v1/wines/664766e5-a73d-5609-a0d7-cffbbe62d466/photos/main.jpg",
                "photos": [
                    {
                        "object_name": "664766e5-a73d-5609-a0d7-cffbbe62d466/main.jpg",
                        "filename": "main.jpg",
                        "url": "/api/v1/wines/664766e5-a73d-5609-a0d7-cffbbe62d466/photos/main.jpg",
                        "is_main": True,
                        "size_bytes": 120000,
                        "content_type": "image/jpeg",
                    }
                ],
                "food_pairings": [],
            }
        }
    )

    id: str
    name: str
    producer: str | None = None
    country: str | None = None
    region: str | None = None
    vintage: int | None = None
    grapes: list[str] = Field(default_factory=list)
    style: str | None = None
    average_rating: float = Field(default=0, ge=0, le=5)
    ratings_count: int = Field(default=0, ge=0)
    price: float | None = Field(default=None, ge=0)
    currency: str | None = None
    image_url: str | None = None
    photos: list[WinePhotoResponse] = Field(default_factory=list)
    description: str | None = None
    rating: float | None = Field(default=None, ge=0, le=5)
    color: str | None = None
    wine_type: str | None = None
    url: str | None = None
    shade: str | None = None
    serving_temperature: str | None = None
    alcohol: str | None = None
    food_pairings: list[str] = Field(default_factory=list)


class ExtractedWineLabel(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Casal Garcia Fruitzy",
                "producer": "Casal Garcia",
                "wine_type": "MELON",
                "region": None,
                "alcohol": "7.5%",
                "vintage": None,
                "label_description": "Natural flavours, 750 ml",
                "volume": 750,
            }
        }
    )

    name: str | None = None
    producer: str | None = None
    wine_type: str | None = None
    region: str | None = None
    alcohol: str | None = None
    vintage: str | None = None
    label_description: str | None = None
    volume: float | None = None


class CompactSearchMatch(BaseModel):
    wine_id: str
    score: float = Field(ge=0, le=1)


class CompactSearchResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "result": [
                    {
                        "wine_id": "664766e5-a73d-5609-a0d7-cffbbe62d466",
                        "score": 0.6544,
                    }
                ]
            }
        }
    )

    result: list[CompactSearchMatch]


class SearchMatch(BaseModel):
    wine: WineResponse
    score: float = Field(ge=0, le=1)
    reason: str
    vector_score: float | None = None
    text_score: float | None = None
    sources: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_id": "4982ce4f-aa67-4ea8-af5f-8cee94e9e83d",
                "filename": "bottle.jpg",
                "content_type": "image/jpeg",
                "image_size_bytes": 215739,
                "recognized_label": "CASAL GARCIA Fruitzy MELON",
                "label_crop": {
                    "box": [481, 1059, 985, 1801],
                    "confidence": 0.9516,
                },
                "extracted_full_image": {
                    "name": "Casal Garcia Fruitzy",
                    "producer": "Casal Garcia",
                    "wine_type": "MELON",
                    "region": None,
                    "alcohol": None,
                    "vintage": None,
                    "label_description": "Natural flavours",
                    "volume": 750,
                },
                "extracted_label_crop": {
                    "name": "Fruitzy",
                    "producer": "CASAL GARCIA",
                    "wine_type": "MELON",
                    "region": None,
                    "alcohol": None,
                    "vintage": None,
                    "label_description": "Natural flavours",
                    "volume": 750,
                },
                "results": [],
            }
        }
    )

    query_id: UUID
    filename: str | None = None
    content_type: str
    image_size_bytes: int = Field(ge=1)
    recognized_label: str | None = None
    label_crop: dict | None = None
    extracted_full_image: ExtractedWineLabel | None = None
    extracted_label_crop: ExtractedWineLabel | None = None
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
