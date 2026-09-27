from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile, status

from src.api.auth import OptionalUserId, ensure_anon_id
from src.api.schemas import SearchResponse, SlugResponse, SommelierSearchResponse, WineResponse
from src.api.services import WineService
from src.api.services.sommelier_service import SORTS, SommelierQuery, SommelierService
from src.api.services.user_service import Owner
from src.api.services.wine_service import SearchOutcome
from src.api.utils import _read_image_upload

router = APIRouter()

IMAGE_ERRORS = {
    400: {"description": "Uploaded image is empty"},
    415: {"description": "Uploaded file is not an image"},
}
PHOTO_CACHE_SECONDS = 7 * 24 * 3600  # фото каталога не меняются


def get_wine_service(request: Request) -> WineService:
    return request.app.state.wine_service


WineServiceDep = Annotated[WineService, Depends(get_wine_service)]
ImageUpload = Annotated[UploadFile, File(description="Фото бутылки или этикетки")]
PhotoSource = Annotated[
    str,
    Query(
        description=(
            "Откуда фото: camera — только что снято кнопкой «Камера», file — галерея, файл, перетаскивание. "
            "Достижения засчитывают только сканы с камеры."
        )
    ),
]
SaveHistory = Annotated[
    bool,
    Query(description="Сохранить поиск в историю (аккаунт по токену или временная история по cookie)"),
]


async def _record_search(
    request: Request,
    response: Response,
    user_id: UUID | None,
    outcome: SearchOutcome,
    source: str = "file",
) -> str | None:
    owner = Owner(user_id=user_id, anon_id=None if user_id else ensure_anon_id(request, response))
    candidates = outcome.candidates
    return await request.app.state.user_service.record_search(
        owner,
        [(candidate.wine_id, candidate.score) for candidate in candidates],
        candidates[0].score if candidates else None,
        from_camera=source == "camera",
    )


@router.post(
    "/search/image",
    response_model=SlugResponse,
    tags=["search"],
    summary="Найти вино по фото",
    description="Возвращает только slug найденного вина. Подробности — `/search/image/extended`.",
    responses=IMAGE_ERRORS,
)
async def search_by_image(
    request: Request,
    response: Response,
    wine_service: WineServiceDep,
    user_id: OptionalUserId,
    image: ImageUpload,
    save_history: SaveHistory = True,
    source: PhotoSource = "file",
) -> SlugResponse:
    outcome = await wine_service.search(await _read_image_upload(image))
    if save_history:
        await _record_search(request, response, user_id, outcome, source)
    return SlugResponse(slug=outcome.candidates[0].slug if outcome.candidates else None)


@router.post(
    "/search/image/extended",
    response_model=SearchResponse,
    tags=["search"],
    summary="Найти вино по фото: результаты с карточками и диагностикой",
    description=(
        "Результаты с разложением скора (`visual_score` + `ocr_score` = `final_score`) и карточками вин "
        "(фото — по `wine.photos[].url`), распознанный текст, кропы и время этапов. "
        "`debug=true` — полная отладка (пул кандидатов, признаки OCR) для офлайн-анализа."
    ),
    responses=IMAGE_ERRORS,
)
async def search_by_image_extended(
    request: Request,
    response: Response,
    wine_service: WineServiceDep,
    user_id: OptionalUserId,
    image: ImageUpload,
    limit: int = Query(default=10, ge=1, le=50, description="Сколько результатов вернуть"),
    main_photos_only: bool = Query(default=False, description="Искать только по главным фото каталога"),
    views: list[str] = Query(
        default=[],
        description="Для экспериментов: искать только по этим кропам (original, bottle_crop, label_crop).",
    ),
    save_history: SaveHistory = True,
    debug: bool = Query(default=False, description="Добавить полную диагностику"),
    source: PhotoSource = "file",
) -> SearchResponse:
    outcome = await wine_service.search(
        await _read_image_upload(image),
        limit=limit,
        views=views or None,
        main_photos_only=main_photos_only,
    )
    search_id = await _record_search(request, response, user_id, outcome, source) if save_history else None
    return await wine_service.build_response(outcome, search_id=search_id, debug=debug)


@router.get(
    "/wines/{wine_id}",
    response_model=WineResponse,
    tags=["wines"],
    summary="Карточка вина",
    responses={404: {"description": "Wine not found"}},
)
async def get_wine(wine_id: str, wine_service: WineServiceDep) -> WineResponse:
    wine = await wine_service.get_wine(wine_id)
    if wine is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wine not found")
    return wine


@router.get(
    "/photos/{photo_id}",
    tags=["wines"],
    summary="Фото вина",
    description="Файл фото из MinIO. Ссылки приходят в `wine.photos[].url`; ответ кешируется браузером.",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}, 404: {"description": "Photo not found"}},
)
async def get_photo(photo_id: str, wine_service: WineServiceDep) -> Response:
    photo = await wine_service.get_photo(photo_id)
    if photo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Photo not found")
    file_data, content_type = photo
    return Response(
        content=file_data.getvalue(),
        media_type=content_type,
        headers={"Cache-Control": f"public, max-age={PHOTO_CACHE_SECONDS}, immutable"},
    )


def get_sommelier_service(request: Request) -> SommelierService:
    return request.app.state.sommelier_service


@router.get(
    "/sommelier/search",
    response_model=SommelierSearchResponse,
    tags=["sommelier"],
    summary="Сомелье: подбор и поиск вина",
    description=(
        "Без нейросети: фильтры по полям каталога, повод (правила по цвету, сахару и гастросочетаниям), "
        "блюда и вкус (ключевые слова описания), поиск по названию/производителю/сорту с исправлением опечаток. "
        "Блюда и вкусы складываются (И), значения одного фильтра — варианты (ИЛИ). "
        "`facets` — все варианты фильтров со счётчиком: сколько вин будет, если выбрать вариант."
    ),
)
async def sommelier_search(
    sommelier: Annotated[SommelierService, Depends(get_sommelier_service)],
    q: Annotated[str, Query(max_length=200, description="Название, производитель, сорт или регион")] = "",
    occasion: Annotated[str | None, Query(description="aperitif, dinner, gift, party, date, picnic, grill, dessert")] = None,
    dish: Annotated[list[str], Query(description="Группы блюд: cheese, fish, meat, ...")] = [],
    taste: Annotated[list[str], Query(description="fresh, fruity, floral, aged, full, light, spicy")] = [],
    type: Annotated[list[str], Query(description="sparkling, still")] = [],
    color: Annotated[list[str], Query()] = [],
    sugar: Annotated[list[str], Query()] = [],
    region: Annotated[list[str], Query()] = [],
    grape: Annotated[list[str], Query()] = [],
    min_rating: Annotated[float | None, Query(ge=0, le=5)] = None,
    sort: Annotated[str, Query(description="relevance, rating, name")] = "relevance",
    limit: Annotated[int, Query(ge=1, le=60)] = 12,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SommelierSearchResponse:
    return await sommelier.search(
        SommelierQuery(
            text=q,
            occasion=occasion,
            dishes=dish,
            tastes=taste,
            types=type,
            colors=color,
            sugars=sugar,
            regions=region,
            grapes=grape,
            min_rating=min_rating,
            sort=sort if sort in SORTS else "relevance",
            limit=limit,
            offset=offset,
        )
    )
