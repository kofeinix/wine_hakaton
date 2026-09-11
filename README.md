# WineHakaton

Локальный demo-stack для аналога Vivino: API, Postgres, Redis и Qdrant запускаются через Docker Compose. Тяжелая LLM для извлечения текста может жить на VPS за API, а локальные CV-веса лежат в `./models` и монтируются в контейнер как `/models`.

## Что лежит локально

- `models/yolo/best.onnx` - YOLO26 ONNX-модель для кропа этикеток. Файл хранится в репо и монтируется в контейнер как `/models/yolo/best.onnx`.
- `models/siglip2/` - внешняя папка с весами `google/siglip2-base-patch16-224`. Веса не входят в Docker image и не должны коммититься.
- Qdrant хранит векторы локально в Docker volume.

## Быстрый запуск

```bash
cp .env.example .env
uv sync
./prepare_models
docker compose up --build
```

API будет доступен на `http://localhost:8000`.

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
Он использует тот же Docker image, что и приложение, запускает `scripts/init_database.py`, создает таблицы через SQLAlchemy-модели из `src/connections/database/models.py` и загружает JSON-экспорт из `data/export`.

При `docker compose up --build` сервис `db-init`:

- ждет healthy `postgres`;
- создает таблицы `wineries` и `wines`;
- валидирует и нормализует `data/export/wineries.json` и `data/export/wines.json`;
- загружает данные через idempotent upsert;
- не сохраняет в `wines` исходные `url`, `photo_url` и пути к фото.

Пути к фото не хранятся в Postgres. Код приложения должен строить MinIO-префикс набора фото из bucket/prefix и `Wine.wine_id`, например `wine/<wine_id>`.

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

Векторы для поиска по фото генерируются отдельным скриптом из главных фотографий вина. Главной считается фотография из `data/photos.tar.gz` с именем `main*.webp` внутри папки конкретного `wine_id`.

Сгенерировать файл эмбеддингов:

```bash
uv run python scripts/generate_wine_embeddings.py \
  --photos-archive data/photos.tar.gz \
  --model-dir models/siglip2 \
  --output data/embeddings/wine_main_siglip2.jsonl.gz
```

Для диагностики производительности добавьте `--profile`; скрипт покажет время чтения/декода из `tar.gz`, preprocessing, переноса на device, inference и записи JSONL:

```bash
uv run python scripts/generate_wine_embeddings.py \
  --photos-archive data/photos.tar.gz \
  --model-dir models/siglip2 \
  --output data/embeddings/wine_main_siglip2.jsonl.gz \
  --batch-size 16 \
  --device auto \
  --profile
```

Файл содержит JSONL-записи с `wine_id`, `main_photo_path` и нормализованным SigLIP2-вектором. Он нужен как воспроизводимый промежуточный артефакт: когда изменится датасет, достаточно пересобрать этот файл и заново выполнить init Qdrant.

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
- читает `/data/embeddings/wine_main_siglip2.jsonl.gz`;
- пересоздает коллекцию `QDRANT__COLLECTION_NAME` (по умолчанию `wine_vectors`);
- записывает точки с id = `wine_id`;
- кладет в payload `wine_id` и `main_photo_path`.

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

## API-заглушки

- `GET /health` - проверка, что API жив.
- `POST /api/v1/search/image` - принимает изображение этикетки в multipart-поле `image` и возвращает список найденных вин. Сейчас это заглушка для будущего OCR/CV/Qdrant-поиска.
- `GET /api/v1/wines` - простой список вин с фильтрами `q`, `country`, `grape`, `min_rating`, `limit`.
- `GET /api/v1/wines/{wine_id}` - детали вина по id.
- `GET /api/v1/wines/{wine_id}/reviews` - отзывы по id вина.
- `POST /api/v1/wines/{wine_id}/reviews` - публикация отзыва.
- `GET /api/v1/dictionaries` - справочники стран, сортов винограда и стилей для фильтров.

Пример поиска по изображению:

```bash
curl -X POST "http://localhost:8000/api/v1/search/image?limit=3" \
  -F "image=@./label.jpg"
```

Пример публикации отзыва:

```bash
curl -X POST "http://localhost:8000/api/v1/wines/wine_001/reviews" \
  -H "Content-Type: application/json" \
  -d '{"author_name":"Alice","rating":4.5,"text":"Good balance and long finish."}'
```

## Подготовить модели

Команда `prepare_models.py` проверяет наличие YOLO ONNX в `models/yolo/best.onnx` и скачивает SigLIP2 в `models/siglip2`.

```bash
./prepare_models
```

Модель `google/siglip2-base-patch16-224` занимает около 1.5 GB. Скачивание нужно сделать один раз; повторный запуск переиспользует уже скачанные файлы.

## YOLO ONNX

В Docker Compose папка `./models` монтируется read-only:

```text
./models:/models:ro
```

Приложение использует дефолтный путь `/models/yolo/best.onnx`. Если модель переэкспортируется из `.pt`, замените только `models/yolo/best.onnx`; зависимость `ultralytics` в runtime не нужна.

## NuExtract / LLM

Приложение работает с NuExtract через OpenAI-compatible Chat Completions API. Сам инференс можно держать на VPS или запустить локально отдельным процессом, а в `.env` приложения указать только endpoint, ключ и served model name.

Минимальная конфигурация для Docker Compose:

```env
LLM__BASE_URL=http://host.docker.internal:1234/v1
LLM__API_KEY=EMPTY
LLM__MODEL_NAME=nuextract3
```

Если NuExtract запущен на VPS, замените `LLM__BASE_URL` на публичный `/v1` endpoint и задайте реальный `LLM__API_KEY` (в рамках хакатона можно запросить у команды ASD):

```env
LLM__BASE_URL=https://vps.example.com/v1
LLM__API_KEY=change-me
LLM__MODEL_NAME=nuextract3
```

`LLM__MODEL_NAME` должен совпадать с именем, под которым модель отдается сервером. В локальных скриптах это имя задается параметром `--served-model-name` или переменной `NUEXTRACT_SERVED_MODEL_NAME`; по умолчанию используется `nuextract3`.

## Локальный запуск NuExtract

Для macOS/Linux используйте корневой скрипт:

```bash
./run_nuextract.sh --device auto
```

Для Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_nuextract.ps1 -Device auto
```

Скрипты поднимают OpenAI-compatible endpoint и печатают готовые значения для `.env`.

Выбор runtime:

- macOS Apple Silicon: `auto` выбирает `mps` и запускает vLLM-Metal с `numind/NuExtract3-mlx-mxfp8`.
- macOS Intel: `auto` выбирает `cpu` и запускает Ollama с `numind/nuextract3:q4_k_m`.
- Linux с NVIDIA GPU: `auto` выбирает `gpu` и запускает vLLM CUDA с `numind/NuExtract3-FP8`.
- Linux без NVIDIA GPU: `auto` выбирает `cpu` и запускает vLLM CPU с `numind/NuExtract3-W8A8`.
- Windows: `auto` выбирает `gpu`, если доступен NVIDIA GPU, иначе `cpu`; оба режима идут через Ollama.

Примеры запуска:

```bash
./run_nuextract.sh mps
./run_nuextract.sh gpu
./run_nuextract.sh cpu --max-model-len 4096
./run_nuextract.sh mps --max-model-len 2048 --served-model-name nuextract3
./run_nuextract.sh gpu --model numind/NuExtract3-FP8 --port 1234
./run_nuextract.sh --device mps --clean
```

```powershell
powershell -ExecutionPolicy Bypass -File .\run_nuextract.ps1 -Device cpu
powershell -ExecutionPolicy Bypass -File .\run_nuextract.ps1 -Device gpu -Port 11434 -ServedModelName nuextract3
```

Порты по умолчанию отличаются по runtime:

- vLLM/vLLM-Metal: `1234`;
- Ollama: `11434`.

Если приложение запущено в Docker, в `.env` нужен адрес Docker-хоста, а не `localhost`:

```env
LLM__BASE_URL=http://host.docker.internal:1234/v1
LLM__API_KEY=EMPTY
LLM__MODEL_NAME=nuextract3
```

Для Ollama на дефолтном порту:

```env
LLM__BASE_URL=http://host.docker.internal:11434/v1
LLM__API_KEY=ollama
LLM__MODEL_NAME=nuextract3
```

Основные параметры `run_nuextract.sh` можно передавать флагами или переменными окружения: `NUEXTRACT_DEVICE`, `NUEXTRACT_HOST`, `NUEXTRACT_PORT`, `NUEXTRACT_MODEL`, `NUEXTRACT_SERVED_MODEL_NAME`, `NUEXTRACT_MAX_MODEL_LEN`, `NUEXTRACT_GPU_MEMORY_UTILIZATION`, `PYTHON_BIN`, `VLLM_VERSION`, `VLLM_METAL_CHANNEL`, `OLLAMA_MODEL`.

## Полезные пути внутри контейнера

- YOLO: `/models/yolo/best.onnx`
- SigLIP2: `/models/siglip2`
- Hugging Face cache: `/models/hf-cache`
