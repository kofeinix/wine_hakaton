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

## LLM на VPS

Для `nuextract3` используйте внешний endpoint:

```env
LLM__BASE_URL=https://your-vps.example.com/v1
LLM__API_KEY=change-me
LLM__MODEL=nuextract3
```

Лучше держать API OpenAI-compatible, чтобы локальный код не зависел от конкретного сервера инференса.

## Полезные пути внутри контейнера

- YOLO: `/models/yolo/best.onnx`
- SigLIP2: `/models/siglip2`
- Hugging Face cache: `/models/hf-cache`
