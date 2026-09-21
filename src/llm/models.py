from pydantic import BaseModel, field_validator

answer_template = {
    "name": "verbatim-string",
    "producer": "verbatim-string",
    "wine_type": "verbatim-string",
    "region": "verbatim-string",
    "alcohol": "verbatim-string",
    "vintage": "verbatim-string",
    "label_description": "string",
    "volume": "number",
}


class WineOutput(BaseModel):
    name: str | None = None
    producer: str | None = None
    wine_type: str | None = None
    region: str | None = None
    alcohol: str | None = None
    vintage: str | None = None
    label_description: str | None = None
    volume: float | None = None

    @field_validator("alcohol", mode="before")
    @classmethod
    def normalize_alcohol(cls, value: object) -> object:
        if isinstance(value, int | float):
            return f"{value:g}%"
        return value


class WineSelectionOutput(BaseModel):
    selected_number: int
    selected_wine_id: str | None = None
    confidence: float | None = None
    reason: str | None = None
