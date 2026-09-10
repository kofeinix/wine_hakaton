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
