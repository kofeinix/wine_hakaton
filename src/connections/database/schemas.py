from decimal import Decimal
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


def _empty_to_none(value: object) -> object:
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _split_csv_list(value: object) -> object:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class WineryBase(BaseModel):
    name: str = Field(min_length=1)
    address: str | None = None
    description: str | None = None
    vineyard_area: str | None = None
    region: str | None = None
    locality: str | None = None
    climate: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def normalize_empty_strings(cls, value: object) -> object:
        return _empty_to_none(value)


class WineryCreate(WineryBase):
    winery_id: UUID


class WineryRead(WineryBase):
    model_config = ConfigDict(from_attributes=True)

    winery_id: UUID


class WineBase(BaseModel):
    winery_id: UUID | None = None
    name: str = Field(min_length=1)
    producer: str | None = None
    rating: Decimal | None = Field(default=None, ge=0, le=5)
    color: str | None = None
    wine_type: str | None = None
    region: str | None = None
    grape_varieties: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("grape_varieties", "grape_variety"),
    )
    shade: str | None = None
    description: str | None = None
    serving_temperature: str | None = None
    alcohol: str | None = None
    food_pairings: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("food_pairings", "food_pairing"),
    )

    @field_validator(
        "winery_id",
        "producer",
        "rating",
        "color",
        "wine_type",
        "region",
        "shade",
        "description",
        "serving_temperature",
        "alcohol",
        mode="before",
    )
    @classmethod
    def normalize_empty_strings(cls, value: object) -> object:
        return _empty_to_none(value)

    @field_validator("grape_varieties", "food_pairings", mode="before")
    @classmethod
    def normalize_lists(cls, value: object) -> object:
        return _split_csv_list(value)


class WineCreate(WineBase):
    wine_id: UUID


class WineRead(WineBase):
    model_config = ConfigDict(from_attributes=True)

    wine_id: UUID
    winery: WineryRead | None = None


class WineListRead(BaseModel):
    items: list[WineRead]
    total: int = Field(ge=0)
