#!/usr/bin/env bash
# Запуск vision-LLM Qwen3-VL-4B для распознавания текста этикеток (OCR).
# LLM необязательна: без неё поиск работает только по картинке, без реранка по тексту.
#
#   ./run_llm.sh              # LM Studio (по умолчанию; macOS / Windows / Linux)
#   ./run_llm.sh ollama       # Ollama
#   ./run_llm.sh vllm         # vLLM на Linux с NVIDIA GPU
#
# После запуска пропишите в .env значения, которые скрипт выведет в конце.
set -Eeuo pipefail

BACKEND="${1:-lmstudio}"
PORT="${LLM_PORT:-1234}"
# на Linux контейнер ходит к хосту через docker-мост, поэтому слушаем все интерфейсы
BIND="127.0.0.1"
[[ "$(uname -s)" == "Linux" ]] && BIND="0.0.0.0"

die() { printf 'ошибка: %s\n' "$*" >&2; exit 1; }
print_env() {
  cat <<EOF

LLM запущена. Пропишите в .env:
  LLM__BASE_URL=http://host.docker.internal:$1/v1
  LLM__MODEL_NAME=$2
и перезапустите приложение: docker compose up -d app
EOF
}

case "$BACKEND" in
  lmstudio)
    LMS="$(command -v lms || true)"
    [[ -z "$LMS" && -x "$HOME/.lmstudio/bin/lms" ]] && LMS="$HOME/.lmstudio/bin/lms"
    [[ -n "$LMS" ]] || die "не найден lms: установите LM Studio (https://lmstudio.ai) и запустите его один раз"
    MODEL="qwen/qwen3-vl-4b"
    "$LMS" get "$MODEL" --yes
    # уже загруженную модель не грузим повторно (иначе LM Studio поднимет вторую копию)
    "$LMS" ps | grep -q "^$MODEL " || "$LMS" load "$MODEL" --context-length 8192 --yes
    "$LMS" server status 2>&1 | grep -q "running" || "$LMS" server start --port "$PORT" --bind "$BIND"
    print_env "$PORT" "$MODEL"
    ;;
  ollama)
    command -v ollama >/dev/null || die "не найден ollama: https://ollama.com/download"
    MODEL="qwen3-vl:4b"
    PORT="${LLM_PORT:-11434}"
    if ! curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null; then
      OLLAMA_HOST="$BIND:$PORT" nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
      for _ in $(seq 1 30); do curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null && break; sleep 1; done
    elif [[ "$BIND" == "0.0.0.0" ]]; then
      echo "Ollama уже запущена; на Linux она должна слушать 0.0.0.0 (OLLAMA_HOST=0.0.0.0), иначе контейнер её не увидит."
    fi
    ollama pull "$MODEL"
    print_env "$PORT" "$MODEL"
    ;;
  vllm)
    [[ "$(uname -s)" == "Linux" ]] || die "vLLM-вариант — для Linux с NVIDIA GPU; на macOS используйте lmstudio или ollama"
    command -v uvx >/dev/null || die "нужен uv: https://docs.astral.sh/uv/"
    MODEL="qwen/qwen3-vl-4b"
    print_env "$PORT" "$MODEL"
    exec uvx --from "vllm>=0.11" vllm serve Qwen/Qwen3-VL-4B-Instruct \
      --host 0.0.0.0 --port "$PORT" --served-model-name "$MODEL" --max-model-len 8192
    ;;
  *)
    die "неизвестный вариант '$BACKEND': lmstudio | ollama | vllm"
    ;;
esac
