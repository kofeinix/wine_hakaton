from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _empty_to_none(value: object) -> object:
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


class ProducerCreate(BaseModel):
    id: UUID
    name: str = Field(min_length=1)


class RegionCreate(BaseModel):
    id: UUID
    name: str = Field(min_length=1)
    country: str | None = None

    @field_validator("country", mode="before")
    @classmethod
    def normalize_empty_strings(cls, value: object) -> object:
        return _empty_to_none(value)


class GrapeCreate(BaseModel):
    id: UUID
    name: str = Field(min_length=1)


class GrapeAliasCreate(BaseModel):
    grape_id: UUID
    alias: str = Field(min_length=1)


class ColorAliasCreate(BaseModel):
    color: str = Field(min_length=1)
    alias: str = Field(min_length=1)


class WineCreate(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=False)

    id: UUID
    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    producer_id: UUID
    region_id: UUID | None = None
    year: int | None = None
    color: str | None = None
    sugar: str | None = None
    alcohol: str | None = None
    price: Decimal | None = None
    stock: int | None = None
    description: str | None = None
    serving_temperature: str | None = None
    shade: str | None = None
    source_url: str | None = None
    created_at: datetime
    updated_at: datetime
    rating: Decimal | None = Field(default=None, ge=0, le=5)

    @field_validator(
        "region_id",
        "year",
        "color",
        "sugar",
        "price",
        "stock",
        "description",
        "serving_temperature",
        "shade",
        "source_url",
        "rating",
        mode="before",
    )
    @classmethod
    def normalize_empty_strings(cls, value: object) -> object:
        return _empty_to_none(value)

    @field_validator("alcohol", mode="before")
    @classmethod
    def normalize_alcohol(cls, value: object) -> object:
        value = _empty_to_none(value)
        if value is None:
            return None
        if isinstance(value, int | float | Decimal):
            number = f"{float(value):g}"
            return f"{number}%"
        return str(value).strip()


class WineGrapeCreate(BaseModel):
    wine_id: UUID
    grape_id: UUID
    percentage: float | None = None

    @field_validator("percentage", mode="before")
    @classmethod
    def normalize_empty_strings(cls, value: object) -> object:
        return _empty_to_none(value)


class FoodCreate(BaseModel):
    id: UUID
    name: str = Field(min_length=1)


class WineFoodCreate(BaseModel):
    id: UUID
    wine_id: UUID
    food_id: UUID


class WineImageCreate(BaseModel):
    id: UUID
    wine_id: UUID
    source_url: str | None = None
    is_main: bool
    is_generated: bool = False
    minio_path: str = Field(min_length=1)
    webp_minio_path: str | None = None

    @field_validator("source_url", mode="before")
    @classmethod
    def normalize_empty_strings(cls, value: object) -> object:
        return _empty_to_none(value)


class WineTermCreate(BaseModel):
    id: UUID
    term: str = Field(min_length=1)
    letter: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    source_url: str | None = None
