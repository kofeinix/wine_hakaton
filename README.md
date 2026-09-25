# WineHakaton

Локальный demo-stack для аналога Vivino: API, Postgres, Redis и Qdrant запускаются через Docker Compose. OCR/vision-модель может жить на VPS за API, а локальные CV-веса лежат в `./models` и монтируются в контейнер как `/models`.

## Что лежит локально

- `models/yolo/label.pt` и `models/yolo/yolo26x.pt` - YOLO PT-модели для кропа этикеток и bbox-кропа бутылки. Файлы монтируются в контейнер как `/models/yolo/*.pt`.
- `models/siglip2/` - внешняя папка с весами `google/siglip2-base-patch16-224`. Веса не входят в Docker image и не должны коммититься.
- Qdrant хранит векторы локально в Docker volume.

## Быстрый запуск

```bash
cp .env.example .env
uv sync
./prepare_models
uv run python scripts/generate_wine_embeddings.py \
  --photos-archive data/photos.tar.gz \
  --images-json data/db/wine_images.json \
  --model-dir models/siglip2 \
  --output data/embeddings/wine_siglip2.jsonl.gz
docker compose up --build
```

API будет доступен на `http://localhost:8000`.

Frontend будет доступен на `http://localhost:5173`.

OpenAPI-документация будет доступна на `http://localhost:8000/docs`.

## Фото в MinIO

Фотографии передаются одним архивом `data/photos.tar.gz`.
При `docker compose up --build` Compose автоматически:

- запускает `minio-init` на базе `minio/mc`;
- создает bucket из `MINIO__BUCKET` (по умолчанию `wine`);
- загружает `data/photos.tar.gz` через MinIO Snowball auto-extract (`X-Amz-Meta-Snowball-Auto-Extract=true`);
- MinIO сам извлекает содержимое архива в bucket отдельными объектами;
- сохраняет структуру `data/photos/<wine_id>/<filename>` как `wine/<wine_id>/<filename>` внутри bucket.

## Postgres init

Начальная загрузка Postgres выполняется отдельным Compose-сервисом `db-init`.
Он использует тот же Docker image, что и приложение, запускает `scripts/init_database.py`, создает таблицы через SQLAlchemy-модели из `src/connections/database/models.py` и загружает нормализованные JSON-файлы из `data/db`.

При `docker compose up --build` сервис `db-init`:

- ждет healthy `postgres`;
- создает таблицы `producers`, `regions`, `grapes`, `wines`, `wine_grapes` и `wine_images`;
- валидирует `data/db/*.json` через Pydantic-схемы;
- загружает данные через idempotent upsert;
- сохраняет в `wines` исходный `url` из `wines.json`;
- не сохраняет в `wines` исходные `photo_url` и пути к фото.

Пути к фото не хранятся в таблице `wines`. Для фотографий есть отдельная таблица `wine_images`: она хранит UUID фотографии, `wine_id`, признак главной фотографии и object key в MinIO.

Чтобы полностью переинициализировать локальную БД, удалите volume:

```bash
docker compose down -v
docker compose up --build
```

Это удобно для демо: жюри достаточно выполнить быстрый запуск, после чего фотографии уже будут лежать в MinIO.
Распакованная копия не хранится ни в Docker volume, ни в init-контейнере.

Запустить только импорт БД повторно можно так:

```bash
docker compose run --rm db-init
```

## Qdrant init

Векторы для поиска по фото генерируются отдельным скриптом для всех фотографий из `data/photos.tar.gz`, которые есть в `data/db/wine_images.json`.

Сгенерировать файл эмбеддингов:

```bash
uv run python scripts/generate_wine_embeddings.py \
  --photos-archive data/photos.tar.gz \
  --images-json data/db/wine_images.json \
  --model-dir models/siglip2 \
  --output data/embeddings/wine_siglip2.jsonl.gz
```

Для диагностики производительности добавьте `--profile`; скрипт покажет время чтения/декода из `tar.gz`, preprocessing, переноса на device, inference и записи JSONL:

```bash
uv run python scripts/generate_wine_embeddings.py \
  --photos-archive data/photos.tar.gz \
  --images-json data/db/wine_images.json \
  --model-dir models/siglip2 \
  --output data/embeddings/wine_siglip2.jsonl.gz \
  --batch-size 16 \
  --device auto \
  --profile
```

Файл содержит JSONL-записи с `photo_id` и нормализованным SigLIP2-вектором. `scripts/init_qdrant.py` принимает только этот новый формат; старый файл `wine_main_siglip2.jsonl.gz` с `wine_id`/`main_photo_path` несовместим и должен быть пересобран. Когда изменится датасет, достаточно пересобрать embeddings-файл и заново выполнить init Qdrant.

После генерации файла `qdrant-init` выполнится автоматически при обычном запуске:

```bash
docker compose up -d --build
```

Запустить только переинициализацию Qdrant можно так:

```bash
docker compose run --rm qdrant-init
```

Сервис `qdrant-init`:

- ждет healthy `qdrant`;
- читает `/data/embeddings/wine_siglip2.jsonl.gz`;
- пересоздает коллекцию `QDRANT__COLLECTION_NAME` (по умолчанию `wine_vectors`);
- записывает точки с id = `photo_id`;
- не кладет MinIO-ссылки в payload Qdrant; API маппит найденные `photo_id` на `wine_id` через Postgres.

Если нужно сохранить существующую коллекцию и только сделать upsert точек:

```bash
docker compose run --rm qdrant-init python scripts/init_qdrant.py --no-recreate
```

Консоль MinIO доступна на `http://localhost:9001`.
Данные для входа по умолчанию:

```text
login: minio
password: miniosecret
bucket: wine
```

Проверка через Docker:

```bash
docker compose run --rm minio-init
docker compose exec minio mc ls local/wine
```

Если нужно пересобрать архив после изменения локальной папки с фото:

```bash
COPYFILE_DISABLE=1 LC_ALL=C tar --exclude='._*' --exclude='.DS_Store' -czf data/photos.tar.gz -C data/photos .
```

## API

- `GET /health` - проверка, что API жив.
- `POST /api/v1/search/image` - принимает изображение бутылки или этикетки в multipart-поле `image` и возвращает компактный результат `{"result": [{"wine_id": "...", "score": 0.0}]}`. Query-параметр `stages` задает pipeline: `global`, `patches`, `llm`. Legacy-значения тоже поддерживаются: `stages=1` = `global`, `stages=2` = `global,patches`. `main_photos_only=true` ограничивает Qdrant-поиск только векторами `photo_id=main`, исключая `yandex_*`.
- `POST /api/v1/search/image/extended` - тот же поиск, но с диагностикой: crop бутылки/этикетки, OCR rerank, полные карточки вин, scores и источники совпадения. Поддерживает тот же `stages`.
- `GET /api/v1/wines/{wine_id}` - детали вина по id.
- `GET /api/v1/wines/{wine_id}/photos/{filename}` - файл фотографии вина из MinIO. URL приходит в `image_url` и `photos[].url` ответа `GET /api/v1/wines/{wine_id}` или extended search.

Старые demo-ручки `GET /api/v1/wines`, `GET/POST /api/v1/wines/{wine_id}/reviews` и `GET /api/v1/dictionaries` удалены вместе с in-memory stub-сервисом.

Пример поиска по изображению:

```bash
curl -X POST "http://localhost:8000/api/v1/search/image?limit=3&stages=global,patches&main_photos_only=true" \
  -F "image=@./label.jpg"
```

Пример расширенного поиска:

```bash
curl -X POST "http://localhost:8000/api/v1/search/image/extended?limit=3&stages=global,patches" \
  -F "image=@./label.jpg"
```

Пример получения карточки вина:

```bash
curl "http://localhost:8000/api/v1/wines/664766e5-a73d-5609-a0d7-cffbbe62d466"
```

## Frontend

Простой React-интерфейс лежит в `frontend/` и запускается отдельным Compose-сервисом.

При обычном запуске:

```bash
docker compose up --build
```

откройте `http://localhost:5173`. Frontend отправляет фото в `POST /api/v1/search/image/extended`, показывает большую карточку лучшего совпадения со всеми пользовательскими параметрами и ниже выводит похожие варианты.

Для доступа к frontend из интернета через проброшенный порт укажите внешний IP или домен в `.env`:

```env
FRONTEND_PORT=5173
VITE_ALLOWED_HOSTS=203.0.113.10,wine.example.com
```

После этого перезапустите frontend:

```bash
docker compose up -d --build frontend
```

Открывайте `http://<ваш-публичный-ip>:5173` или домен, если порт 5173 проброшен на эту машину. API вызывается через относительный `/api`, поэтому наружу достаточно публиковать frontend-порт; Vite сам проксирует запросы к `app:8000` внутри Docker Compose.

Для локальной frontend-разработки без Docker:

```bash
cd frontend
npm install
npm run dev
```

Vite проксирует `/api` на `http://localhost:8000`. Если API запущен по другому адресу, задайте:

```bash
VITE_API_TARGET=http://localhost:8000 npm run dev
```

## Подготовить модели

Команда `prepare_models.py` проверяет наличие YOLO PT-моделей в `models/yolo`,
скачивает SigLIP2 в `models/siglip2` и DINOv3 в `models/dinov3`.

```bash
./prepare_models
```

Модель `facebook/dinov3-vitb16-pretrain-lvd1689m` закрыта gated-доступом на
Hugging Face. Перед скачиванием примите условия модели в Hugging Face и
передайте токен:

```bash
HF_TOKEN=<your_token> ./prepare_models
```

Модель `google/siglip2-base-patch16-224` занимает около 1.5 GB. Скачивание нужно сделать один раз; повторный запуск переиспользует уже скачанные файлы.

Global-вектора для Qdrant можно экспортировать как SigLIP2 или DINOv3 NPZ:

```bash
uv run python scripts/index_siglip2_views_qdrant.py --encoder siglip2
uv run python scripts/index_siglip2_views_qdrant.py --encoder dinov3
```

Для расширенного набора фото можно сначала досчитать `bottle_crop` и
`label_crop` только для нужных photo-папок, а затем экспортировать эмбеддинги
по тем же маскам:

```bash
uv run python scripts/prepare_image_views.py \
  --images-dir data/images_extended \
  --photo-dir-pattern main 'yandex_*' 'flux_*' 'vivino_*'

# Повторный прогон только по проблемным папкам, по одной папке wine_id/photo_id на строку.
uv run python scripts/prepare_image_views.py \
  --images-dir data/images_extended \
  --photo-dir-file missing_photo_dirs.txt \
  --overwrite

uv run python scripts/prepare_image_views.py \
  --images-dir data/images_extended \
  --photo-dir-file missing_label_dirs.txt \
  --crop-kind label \
  --overwrite

uv run python scripts/prepare_image_views.py \
  --images-dir data/images_extended \
  --photo-dir-file missing_bottle_dirs.txt \
  --crop-kind bottle \
  --overwrite

uv run python scripts/index_siglip2_views_qdrant.py \
  --images-dir data/images_extended \
  --encoder siglip2 \
  --photo-dir-pattern main 'yandex_*' 'flux_*' 'vivino_*'

uv run python scripts/init_qdrant.py --embeddings-dir data/embeddings
```

По умолчанию DINOv3 создаёт collection metadata `wine_original_dinov3`, `wine_label_crop_dinov3`. Для быстрого A/B через текущие runtime collection names можно перезаписать suffix:

```bash
uv run python scripts/index_siglip2_views_qdrant.py --encoder dinov3 --collection-encoder siglip2
uv run python scripts/init_qdrant.py --embeddings-dir data/embeddings
```

Чтобы приложение считало query-вектора той же моделью, что лежит в Qdrant, задайте runtime encoder:

```env
SEARCH__GLOBAL_ENCODER=dinov3
SEARCH__COLLECTION_ENCODER=dinov3
```

Если вы сгенерировали DINOv3-вектора с `--collection-encoder siglip2`, то для A/B через старые имена коллекций используйте:

```env
SEARCH__GLOBAL_ENCODER=dinov3
SEARCH__COLLECTION_ENCODER=siglip2
```

## YOLO PT

В Docker Compose папка `./models` монтируется read-only:

```text
./models:/models:ro
```

Приложение использует дефолтные пути `/models/yolo/label.pt` и `/models/yolo/yolo26x.pt`. Кропы строятся через `ultralytics.YOLO`, без отдельного decode-кода.

## OCR / LLM

Приложение работает с OpenAI-compatible Chat Completions API. В базовом flow vision-модель используется для OCR по приоритету `label_crop -> bottle_crop -> original`, затем OCR-текст участвует в rerank top-50 кандидатов. Отдельный stage `llm` сохранен как опциональный rerank-кандидат, но по умолчанию frontend и API используют только `global,patches`.

### OCR rerank

Код: [`src/ml/ocr_matching.py`](src/ml/ocr_matching.py) (алгоритм и веса),
[`src/api/services/ocr_match_service.py`](src/api/services/ocr_match_service.py) (словарь каталога из БД).
Одна и та же функция используется в API и при сборке датасета CatBoost.

```text
final = visual_score + ocr_bonus
ocr_bonus = 0.1 * (0.5·B(name) + 0.85·B(producer) + 0.25·B(grapes))
          + 0.005 * unique_evidence
          - 0.02·color_contra - 0.02·sugar_contra + 0.01·sugar_match
          - 0.03·grape_contra - 0.05·producer_contra
          + 0.02·(alcohol_match - alcohol_mismatch)
```

- **B(x)** — бонус за сходство сущности выше порога `0.55`. Сходство — IDF-coverage: какая доля
  слов сущности (вес = IDF слова по каталогу × длина) нашлась в OCR; порядок слов не важен.
- **unique_evidence** — сумма редкостей (IDF по пулу из 50 кандидатов) слов названия и
  производителя, найденных в OCR. Слово, общее для всей линейки («Ркацители»), весит 0, а
  «десертное» или «Belmas» у одного кандидата — много. Главный сигнал, различающий вина одного
  производителя.
- **Противоречия** — на этикетке найден другой цвет / сахар / сорт / производитель каталога.
- Отсутствие чего-либо в OCR **не штрафуется**: в формуле только бонусы и явные противоречия.
- Все строки сравниваются как есть и в RU→LAT / LAT→RU транслитерации (и сущности, и OCR).

Диагностика `/search/image/extended` → `diagnostics.ocr_rerank`: `text` (сырой OCR),
`candidate_scores` (top-10 со всеми компонентами) и `pool` (весь пул до реранка:
`visual_score`, `ocr_bonus`) — для офлайн-анализа.

Эксперименты — [`scripts/ocr_lab`](scripts/ocr_lab/README.md). Кросс-валидация на 578 eval-фото
(группировка по вину):

| Вариант | Acc@1 | MRR |
|---|---|---|
| только изображение | 79.4% | 0.876 |
| прежняя формула (0.8·visual + 0.2·ocr) | 86.2% | 0.922 |
| **текущая формула** | **87.4%** (88.2% на всём наборе) | **0.930** |

Из оставшихся ошибок почти все — вина того же производителя, где в OCR нет ни одного слова в
пользу правильного (дубликаты и почти одинаковые карточки в БД).

### CatBoost (этап 2)

`scripts/train_catboost_reranker.py` обучает CatBoost **поверх формулы этапа 1**: скор формулы
передаётся как `baseline` (`--stage1-baseline-scale`, по умолчанию 100), модель учит поправку.
Масштаб пишется в метаданные модели и применяется при инференсе. С нуля CatBoost на ~500
запросах переобучается и проигрывает формуле (81–82% CV). Датасет сохраняет сырой OCR-текст.
Модель, обученную на прежних признаках, нужно переобучить.

Если YOLO не нашел `label_crop` ни на исходном изображении, ни внутри `bottle_crop`, API считает, что пользователь мог прислать близкое фото этикетки: `label_crop` становится равен `original`, и поиск идет по label collection.

Минимальная конфигурация для Docker Compose:

```env
LLM__BASE_URL=http://host.docker.internal:1234/v1
LLM__API_KEY=EMPTY
LLM__MODEL_NAME=glm-ocr
```

Если модель запущена на VPS, замените `LLM__BASE_URL` на публичный `/v1` endpoint и задайте реальный `LLM__API_KEY`:

```env
LLM__BASE_URL=https://vps.example.com/v1
LLM__API_KEY=change-me
LLM__MODEL_NAME=glm-ocr
```

Если приложение запущено в Docker, в `.env` нужен адрес Docker-хоста, а не `localhost`:

```env
LLM__BASE_URL=http://host.docker.internal:1234/v1
LLM__API_KEY=EMPTY
LLM__MODEL_NAME=glm-ocr
```

`LLM__MODEL_NAME` должен совпадать с именем, под которым модель отдается сервером.

## Полезные пути внутри контейнера

- YOLO label: `/models/yolo/label.pt`
- YOLO bottle detection: `/models/yolo/yolo26x.pt`
- SigLIP2: `/models/siglip2`
- Hugging Face cache: `/models/hf-cache`
