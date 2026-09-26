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

Фото лежат в bucket `wine` как `<wine_id>/<photo_dir>.jpg` (так их ищет API по
`wine_images.minio_path`):

- настоящие — `main`, `yandex_*`, `vivino_*` → в API `wine.photos` (`main` первым, он же `image_url`);
- сгенерированные сцены `flux_*` (`wine_images.is_generated = true`) → `wine.generated_photos`,
  фронтенд показывает их по кнопке.

**Свежий запуск** (склонировали репозиторий): положите архив `photos.tar.gz` в `./data/` и
выполните `docker compose up --build`. Сервис `minio-init`:

- если bucket пуст — распаковывает `data/photos.tar.gz` в bucket (Snowball auto-extract);
- если bucket уже заполнен — ничего не делает (`MINIO_RESEED=1 docker compose up minio-init` — перезалить);
- если архива нет — пишет предупреждение и не блокирует запуск (фото будут отдавать 404).

`db-init` при каждом запуске приводит каталог к `data/db/*.json` (`--prune`): лишние строки
каталога удаляются, пользователи, история, избранное и отзывы не трогаются.

Архив собирается ровно по `data/db/wine_images.json` (~5.3 GB; без flux — ~1.2 GB):

```bash
uv run python scripts/build_photos_archive.py                 # все фото
uv run python scripts/build_photos_archive.py --no-generated  # без сгенерированных
```

Обновить набор фото из `data/images_extended` (загрузка в работающий MinIO, синхронизация
`wine_images` и `data/db/wine_images.json`, удаление лишних объектов из bucket):

```bash
export MINIO__HOST=localhost DATABASE__HOST=localhost
uv run python scripts/seed_minio_photos.py                                                  # загрузить новые
uv run python scripts/seed_minio_photos.py --skip-upload --sync-db --prune-minio --dry-run  # посмотреть
uv run python scripts/seed_minio_photos.py --skip-upload --sync-db --prune-minio --write-json
```

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
- `POST /api/v1/search/image` — фото бутылки или этикетки в multipart-поле `image`; ответ — только
  slug найденного вина: `{"slug": "..."}` (`null`, если ничего не найдено).
- `POST /api/v1/search/image/extended` — тот же поиск, ответ для фронтенда:

  ```json
  {
    "search_id": "…",                 // для POST /favorites
    "status": "found",
    "results": [
      {"rank": 1, "slug": "…", "wine_id": "…",
       "visual_score": 0.81, "ocr_score": 0.22, "final_score": 1.03,
       "wine": {"name": "…", "producer": "…", "color": "…", "sugar": "…", "grapes": [],
                "image_url": "/api/v1/photos/<id>", "photos": [{"id": "…", "url": "/api/v1/photos/<id>", "is_main": true}]}}
    ],
    "ocr": {"applied": true, "source_view": "label_crop", "text": "…"},
    "crops": {"original": {…}, "bottle_crop": {…}, "label_crop": {"available": true, "box": [x1, y1, x2, y2]}},
    "search": {"active_views": ["label_crop"], "label_photo_mode": true, "candidates": 50},
    "timings_ms": {"total_ms": 1602, "crops_ms": 113, "ocr_ms": 1310, "embedding_ms": 78,
                   "vector_search_ms": 28, "rerank_ms": 72, "llm_ms": null},
    "diagnostics": null               // полная отладка только при debug=true
  }
  ```

  Параметры: `limit`, `stages` (`global` — всегда; `llm` — дополнительно выбор vision-LLM),
  `main_photos_only`, `views` (эксперименты по кропам), `save_history`, `debug`.
- `GET /api/v1/wines/{wine_id}` — карточка вина.
- `GET /api/v1/photos/{photo_id}` — файл фото из MinIO (ссылки — в `wine.photos[].url` и
  `wine.image_url`), с `Cache-Control` на неделю.

Старые demo-ручки `GET /api/v1/wines`, `GET/POST /api/v1/wines/{wine_id}/reviews` и `GET /api/v1/dictionaries` удалены вместе с in-memory stub-сервисом.

Пример поиска по изображению:

```bash
curl -X POST "http://localhost:8000/api/v1/search/image" \
  -F "image=@./label.jpg"
```

Пример расширенного поиска:

```bash
curl -X POST "http://localhost:8000/api/v1/search/image/extended?limit=3" \
  -F "image=@./label.jpg"
```

Пример получения карточки вина:

```bash
curl "http://localhost:8000/api/v1/wines/664766e5-a73d-5609-a0d7-cffbbe62d466"
```

## Пользователи: вход, история, избранное, напоминания

Упрощённая авторизация: email + пароль (bcrypt), JWT access-токен в заголовке
`Authorization: Bearer <token>` (в Swagger — кнопка **Authorize**). Таблицы `users`,
`search_history`, `favorites`, `wine_reviews`, `notifications`, `notification_wines`
создаются автоматически при старте API.

| Метод | Путь | Доступ | Что делает |
|---|---|---|---|
| POST | `/api/v1/auth/register` | все | регистрация → токен; временная история переносится в аккаунт |
| POST | `/api/v1/auth/login` | все | вход → токен; временная история переносится в аккаунт |
| GET | `/api/v1/auth/me` | токен | текущий пользователь |
| GET | `/api/v1/history` | все | история поиска: аккаунт или временная по cookie `wine_anon_id` |
| GET | `/api/v1/history/wines?since=&until=` | токен | просмотренные вина за промежуток (по умолчанию сутки) с избранным и отзывом |
| DELETE | `/api/v1/history/{search_id}` | все | удалить запись истории |
| GET / POST | `/api/v1/favorites` | токен | список / добавить `{wine_id, search_id?}` |
| DELETE | `/api/v1/favorites/{wine_id}` | токен | убрать из избранного |
| GET | `/api/v1/reviews` | токен | мои оценки и комментарии |
| PUT / DELETE | `/api/v1/reviews/{wine_id}` | токен | оценка 1–5 и/или комментарий `{rating?, comment?, notification_id?}` / удалить |
| GET | `/api/v1/notifications?unread_only=` | токен | напоминания + число непрочитанных |
| GET | `/api/v1/notifications/{id}` | токен | открыть: вина за период с избранным и отзывом (отмечает прочитанным) |
| POST | `/api/v1/notifications/{id}/read`, `/read-all` | токен | отметить прочитанными |

- `/search/image` и `/search/image/extended` сохраняют поиск в историю; extended возвращает
  `search_id` (его можно передать в `POST /favorites`). Без токена поиск работает, история
  временная — по cookie, хранится `AUTH__ANON_HISTORY_TTL_HOURS`. `save_history=false` —
  не сохранять (так делает `scripts/eval_search.py`).
- **Напоминание — одно на сессию.** Сессия закончена, если с последнего поиска прошло
  `NOTIFICATIONS__DELAY_MINUTES`. Тогда все её поиски собираются в одно уведомление «Вы недавно
  смотрели вина (N). Что-то взяли?»: вина (top-1 каждого поиска, без повторов) за последние
  `NOTIFICATIONS__MAX_AGE_HOURS`, которые пользователь ещё не добавил в избранное и не оценил.
  Если таких нет — уведомления нет. По нажатию фронтенд открывает `GET /notifications/{id}`:
  список вин с кнопками «в избранное» (`POST /favorites`) и «оценить» (`PUT /reviews/{wine_id}`
  с `notification_id`). Фоновая задача проверяет раз в `NOTIFICATIONS__CHECK_INTERVAL_SECONDS`.
- Настройки — `AUTH__*` и `NOTIFICATIONS__*` в `.env` (см. `.env.example`). Обязательно задайте
  `AUTH__JWT_SECRET`.

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
и скачивает SigLIP2 в `models/siglip2`.

```bash
./prepare_models
```

Модель `google/siglip2-base-patch16-224` занимает около 1.5 GB. Скачивание нужно сделать один раз; повторный запуск переиспользует уже скачанные файлы.

Global-вектора SigLIP2 для Qdrant экспортируются в NPZ:

```bash
uv run python scripts/index_siglip2_views_qdrant.py
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
  --photo-dir-pattern main 'yandex_*' 'flux_*' 'vivino_*'

uv run python scripts/init_qdrant.py --embeddings-dir data/embeddings
```

## YOLO PT

В Docker Compose папка `./models` монтируется read-only:

```text
./models:/models:ro
```

Приложение использует дефолтные пути `/models/yolo/label.pt` и `/models/yolo/yolo26x.pt`. Кропы строятся через `ultralytics.YOLO`, без отдельного decode-кода.

## OCR / LLM

Приложение работает с OpenAI-compatible Chat Completions API. В базовом flow vision-модель используется для OCR по приоритету `label_crop -> bottle_crop -> original`, затем OCR-текст участвует в rerank top-50 кандидатов. Отдельный stage `llm` — опциональный выбор кандидата vision-LLM; по умолчанию frontend и API используют только `global`.

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

В ответе `/search/image/extended` у каждого результата `visual_score`, `ocr_score` и `final_score`,
распознанный текст — в `ocr.text`. С `debug=true` в `diagnostics.ocr_rerank` добавляются
`candidate_scores` (top-10 со всеми компонентами) и `pool` (весь пул до реранка) — для офлайн-анализа.

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

CatBoost обучается **поверх формулы этапа 1**: её скор идёт в `baseline`
(`--stage1-baseline-scale`, по умолчанию 100), модель учит поправку; масштаб пишется в
метаданные модели и применяется при инференсе. С нуля на ~500 запросах CatBoost
переобучается (81–82% CV против 87.4% у формулы), поэтому обучаем на `images_extended`,
а проверяем на реальном `data/eval`.

Запросы для обучения — `data/images_extended/<wine_id>/<photo>/original.jpg`:

| Папки | Что это | В индексе Qdrant | Как ищем |
|---|---|---|---|
| `yandex_*` | реальные фото из веба | да | leave-one-out: точки этого фото исключаются фильтром |
| `flux_*` | сгенерированные сцены | да | leave-one-out |
| `main` | каталожное фото | да | не бывает запросом |

Leave-one-out оставляет в индексе остальные фото вина — как у реального пользователя, чьё
вино в каталоге есть. Индекс переиндексировать не нужно: признаки считаются на том же индексе,
что и в проде.

```bash
# хосты из .env — докерные; для запуска с хоста:
export QDRANT__HOST=localhost DATABASE__HOST=localhost LLM__BASE_URL=http://localhost:1234/v1

# 1. датасеты (дописываются: сбор можно прервать и продолжить; OCR кешируется в data/catboost/ocr_cache.jsonl)
uv run python scripts/train_catboost_reranker.py collect --source eval --with-ocr
uv run python scripts/train_catboost_reranker.py collect --source extended --with-ocr \
    --max-per-wine 'yandex_*=2' 'flux_*=2'          # по умолчанию yandex_* и flux_*

# 2. обучение на extended, оценка на реальном eval, финальная модель — на extended + eval
uv run python scripts/train_catboost_reranker.py train --refit-with-valid \
    --train-dataset data/catboost/extended.jsonl --valid-dataset data/catboost/eval.jsonl
```

Отчёт печатает Acc@1 / MRR для `baseline` (только изображение), `stage1 formula` и `catboost`
на valid. Число деревьев выбирается early stopping'ом по valid, поэтому метрика catboost на
eval немного оптимистична.

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
