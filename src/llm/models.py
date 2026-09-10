from typing import Literal

from pydantic import BaseModel

answer_template = {
    "winery": "verbatim-string",
    "wine_name": "verbatim-string",
    "manufacture_date": "date",

    "bottle_color": ["Green", "Dark Green", "Amber", "Clear", "Brown", "Blue", "Other"],
    "bottle_shape": ["Bordeaux", "Burgundy", "Tall", "Round", "Standard", "Other"],

    "label_description": "string",

    "wine_type": ["Red", "White", "Rosé", "Sparkling", "Dessert", "Other"],
    "alcohol_amount": "number",
    "volume": "number"
}

class WineOutput(BaseModel):
    winery: str | None = None
    wine_name: str | None = None
    manufacture_date: str | None = None
    bottle_color: Literal["Green", "Dark Green", "Amber", "Clear", "Brown", "Blue", "Other"] | None = None
    bottle_shape: Literal["Bordeaux", "Burgundy", "Tall", "Round", "Standard", "Other"] | None = None
    label_description: str | None = None
    wine_type: Literal["Red", "White", "Rosé", "Sparkling", "Dessert", "Other"] | None = None
    alcohol_amount: float | None = None
    volume: float | None = None
