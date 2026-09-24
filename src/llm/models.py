from pydantic import BaseModel


class WineSelectionOutput(BaseModel):
    selected_number: int
    selected_wine_id: str | None = None
    confidence: float | None = None
    reason: str | None = None
