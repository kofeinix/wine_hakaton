from datetime import UTC, datetime
from uuid import uuid4

from src.api.schemas import (
    DictionaryResponse,
    ReviewCreate,
    ReviewListResponse,
    ReviewResponse,
    SearchMatch,
    SearchResponse,
    WineListResponse,
    WineResponse,
)
from src.container.manager import ConnectionManager


class WineCatalogService:
    """In-memory stub. Replace methods with Postgres/Qdrant/ML-backed logic."""

    def __init__(self, connection_manager: ConnectionManager) -> None:
        self.connection_manager = connection_manager
        self._wines: dict[str, WineResponse] = {
            "wine_001": WineResponse(
                id="wine_001",
                name="Chateau Demo Rouge",
                producer="Demo Estate",
                country="France",
                region="Bordeaux",
                vintage=2019,
                grapes=["Merlot", "Cabernet Sauvignon"],
                style="Red",
                average_rating=4.2,
                ratings_count=128,
                price=24.9,
                currency="EUR",
                image_url="https://example.com/wines/wine_001.jpg",
                description="Dry red wine with blackcurrant, plum and oak notes.",
            ),
            "wine_002": WineResponse(
                id="wine_002",
                name="Hackathon Reserva",
                producer="Prototype Cellars",
                country="Spain",
                region="Rioja",
                vintage=2020,
                grapes=["Tempranillo"],
                style="Red",
                average_rating=4.0,
                ratings_count=76,
                price=18.5,
                currency="EUR",
                image_url="https://example.com/wines/wine_002.jpg",
                description="Medium-bodied red wine with cherry and spice profile.",
            ),
            "wine_003": WineResponse(
                id="wine_003",
                name="Label Finder Sauvignon Blanc",
                producer="Vision Winery",
                country="New Zealand",
                region="Marlborough",
                vintage=2022,
                grapes=["Sauvignon Blanc"],
                style="White",
                average_rating=4.1,
                ratings_count=54,
                price=16.0,
                currency="EUR",
                image_url="https://example.com/wines/wine_003.jpg",
                description="Fresh white wine with citrus, gooseberry and herbal notes.",
            ),
        }
        self._reviews: dict[str, list[ReviewResponse]] = {
            "wine_001": [
                ReviewResponse(
                    id="review_001",
                    wine_id="wine_001",
                    author_name="Demo User",
                    rating=4.5,
                    text="Good balance and a long finish.",
                    created_at=datetime(2026, 9, 10, 12, 0, tzinfo=UTC),
                )
            ]
        }

    def search_by_image(
        self,
        image_bytes: bytes,
        filename: str | None,
        content_type: str,
        limit: int,
    ) -> SearchResponse:
        wines = list(self._wines.values())[:limit]
        matches = [
            SearchMatch(
                wine=wine,
                score=max(0.5, 0.95 - index * 0.08),
                reason="Stub match. Replace with OCR/CV embedding search.",
            )
            for index, wine in enumerate(wines)
        ]
        return SearchResponse(
            query_id=uuid4(),
            filename=filename,
            content_type=content_type,
            image_size_bytes=len(image_bytes),
            recognized_label=None,
            results=matches,
        )

    def list_wines(
        self,
        q: str | None,
        country: str | None,
        grape: str | None,
        min_rating: float | None,
        limit: int,
    ) -> WineListResponse:
        wines = list(self._wines.values())
        if q:
            needle = q.casefold()
            wines = [
                wine
                for wine in wines
                if needle in wine.name.casefold() or needle in wine.producer.casefold()
            ]
        if country:
            wines = [wine for wine in wines if wine.country.casefold() == country.casefold()]
        if grape:
            wines = [
                wine
                for wine in wines
                if any(item.casefold() == grape.casefold() for item in wine.grapes)
            ]
        if min_rating is not None:
            wines = [wine for wine in wines if wine.average_rating >= min_rating]

        return WineListResponse(items=wines[:limit], total=len(wines))

    def get_wine(self, wine_id: str) -> WineResponse | None:
        return self._wines.get(wine_id)

    def get_reviews(self, wine_id: str) -> ReviewListResponse:
        reviews = self._reviews.get(wine_id, [])
        return ReviewListResponse(items=reviews, total=len(reviews))

    def create_review(self, wine_id: str, payload: ReviewCreate) -> ReviewResponse:
        review = ReviewResponse(
            id=f"review_{uuid4().hex}",
            wine_id=wine_id,
            author_name=payload.author_name,
            rating=payload.rating,
            text=payload.text,
            created_at=datetime.now(UTC),
        )
        self._reviews.setdefault(wine_id, []).append(review)
        return review

    def get_dictionaries(self) -> DictionaryResponse:
        wines = self._wines.values()
        countries = sorted({wine.country for wine in wines})
        grapes = sorted({grape for wine in wines for grape in wine.grapes})
        styles = sorted({wine.style for wine in wines if wine.style is not None})
        return DictionaryResponse(countries=countries, grapes=grapes, styles=styles)
