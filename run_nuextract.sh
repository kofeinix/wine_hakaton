#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEVICE="${NUEXTRACT_DEVICE:-auto}"
HOST="${NUEXTRACT_HOST:-0.0.0.0}"
PORT="${NUEXTRACT_PORT:-}"
PORT_SET=false
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VLLM_VERSION="${VLLM_VERSION:-0.28.0}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:-.venv-vllm}"
VLLM_METAL_VENV_DIR="${VLLM_METAL_VENV_DIR:-.venv-vllm-metal}"
VLLM_METAL_CHANNEL="${VLLM_METAL_CHANNEL:-dev}"
OLLAMA_MODEL="${OLLAMA_MODEL:-numind/nuextract3:q4_k_m}"
MODEL="${NUEXTRACT_MODEL:-}"
MODEL_CACHE_ROOT="${NUEXTRACT_MODEL_CACHE_ROOT:-$ROOT_DIR/models/hf-vllm}"
SERVED_MODEL_NAME="${NUEXTRACT_SERVED_MODEL_NAME:-nuextract3}"
MAX_MODEL_LEN="${NUEXTRACT_MAX_MODEL_LEN:-4000}"
GPU_MEMORY_UTILIZATION="${NUEXTRACT_GPU_MEMORY_UTILIZATION:-}"
EXTRA_ARGS=()
CLEAN=false

usage() {
  cat <<'USAGE'
Usage:
  ./run_nuextract.sh [cpu|mps|gpu] [options]
  ./run_nuextract.sh --device cpu|mps|gpu|auto [options]

Platform policy:
  macOS Apple Silicon: mps via vLLM-Metal, cpu via Ollama.
  macOS Intel: cpu via Ollama.
  Linux: gpu via vLLM CUDA, cpu via vLLM CPU.

Options:
  --device VALUE          cpu, mps, gpu, or auto.
  --host HOST            Server host. Default: 0.0.0.0
  --port PORT            Server port. Default: 1234
  --model MODEL          Override model id.
  --served-model-name N  Served OpenAI model name. Default: nuextract3
  --max-model-len N      vLLM context limit. Default: 4000
  --python PYTHON        Python used for venv creation. Default: python3.12
  --vllm-version V       vLLM version for Linux. Default: 0.28.0
  --extra ARG            Pass one extra argument to vLLM serve. Repeatable.
  --clean                Remove selected vLLM venv and exit.
  -h, --help             Show this help.

Environment:
  OLLAMA_MODEL=numind/nuextract3:q4_k_m
  VLLM_CPU_WHEEL_URL=https://.../vllm-...+cpu-...whl
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '[nuextract] %s\n' "$*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    cpu|mps|gpu|auto)
      DEVICE="$1"
      shift
      ;;
    --device)
      [[ $# -ge 2 ]] || die "--device requires a value"
      DEVICE="$2"
      shift 2
      ;;
    --host)
      [[ $# -ge 2 ]] || die "--host requires a value"
      HOST="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || die "--port requires a value"
      PORT="$2"
      PORT_SET=true
      shift 2
      ;;
    --model)
      [[ $# -ge 2 ]] || die "--model requires a value"
      MODEL="$2"
      shift 2
      ;;
    --served-model-name)
      [[ $# -ge 2 ]] || die "--served-model-name requires a value"
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --max-model-len)
      [[ $# -ge 2 ]] || die "--max-model-len requires a value"
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --vllm-version)
      [[ $# -ge 2 ]] || die "--vllm-version requires a value"
      VLLM_VERSION="$2"
      shift 2
      ;;
    --extra)
      [[ $# -ge 2 ]] || die "--extra requires a value"
      EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --clean)
      CLEAN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "$DEVICE" in
  auto|cpu|mps|gpu) ;;
  *) die "--device must be one of: auto, cpu, mps, gpu" ;;
esac

OS_NAME="$(uname -s)"
MACHINE="$(uname -m)"

if [[ "$DEVICE" == "auto" ]]; then
  if [[ "$OS_NAME" == "Darwin" && "$MACHINE" == "arm64" ]]; then
    DEVICE="mps"
  elif [[ "$OS_NAME" == "Darwin" ]]; then
    DEVICE="cpu"
  elif [[ "$OS_NAME" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DEVICE="gpu"
  else
    DEVICE="cpu"
  fi
fi

validate_platform() {
  case "$OS_NAME:$MACHINE:$DEVICE" in
    Darwin:arm64:cpu) return ;;
    Darwin:arm64:mps) return ;;
    Darwin:x86_64:cpu) return ;;
    Linux:*:cpu) return ;;
    Linux:*:gpu)
      command -v nvidia-smi >/dev/null 2>&1 || die "gpu mode needs NVIDIA drivers and nvidia-smi"
      nvidia-smi -L >/dev/null 2>&1 || die "gpu mode was requested, but no NVIDIA GPU is available"
      return
      ;;
  esac

  if [[ "$OS_NAME" == "Darwin" && "$MACHINE" == "arm64" ]]; then
    die "Apple Silicon launcher supports cpu through Ollama and mps through vLLM-Metal. Requested: $DEVICE"
  fi
  if [[ "$OS_NAME" == "Darwin" && "$MACHINE" == "x86_64" ]]; then
    die "Mac Intel launcher supports cpu through Ollama. Requested: $DEVICE"
  fi
  if [[ "$OS_NAME" == "Linux" ]]; then
    die "Linux launcher supports cpu and gpu. Requested: $DEVICE"
  fi
  die "unsupported OS/architecture: $OS_NAME $MACHINE"
}

wait_http() {
  local url="$1"
  local timeout_seconds="${2:-60}"
  local started_at
  started_at="$(date +%s)"

  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - started_at >= timeout_seconds )); then
      return 1
    fi
    sleep 1
  done
}

run_ollama() {
  [[ "$DEVICE" == "cpu" ]] || die "Ollama path only supports cpu in this launcher"

  if [[ "$PORT_SET" == false && -z "${NUEXTRACT_PORT:-}" ]]; then
    PORT="11434"
  fi
  local listen_host="$HOST"
  if [[ "$listen_host" == "0.0.0.0" ]]; then
    listen_host="127.0.0.1"
  fi
  local ollama_host="${OLLAMA_HOST:-$HOST:$PORT}"
  local local_base_url="http://$listen_host:$PORT"

  if ! command -v ollama >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Ollama is required on this machine.

Install it from:
  https://ollama.com/download

Then rerun:
  ./run_nuextract.sh cpu
EOF
    exit 1
  fi

  local tags_url="$local_base_url/api/tags"
  local started_pid=""

  if ! curl -fsS "$tags_url" >/dev/null 2>&1; then
    log "starting Ollama server"
    OLLAMA_HOST="$ollama_host" ollama serve &
    started_pid="$!"
    trap '[[ -n "${started_pid:-}" ]] && kill "$started_pid" >/dev/null 2>&1 || true' EXIT
    wait_http "$tags_url" 60 || die "Ollama did not become ready at $tags_url"
  fi

  log "pulling/checking Ollama model: $OLLAMA_MODEL"
  ollama pull "$OLLAMA_MODEL"

  local modelfile
  modelfile="$(mktemp)"
  cat >"$modelfile" <<EOF
FROM $OLLAMA_MODEL
PARAMETER num_ctx $MAX_MODEL_LEN
EOF
  log "creating Ollama alias: $SERVED_MODEL_NAME -> $OLLAMA_MODEL"
  ollama create "$SERVED_MODEL_NAME" -f "$modelfile"
  rm -f "$modelfile"

  cat <<EOF
[nuextract] Ollama OpenAI-compatible endpoint is ready:
  base_url: $local_base_url/v1
  model:    $SERVED_MODEL_NAME

For Docker on macOS, use:
  LLM__BASE_URL=http://host.docker.internal:$PORT/v1
  LLM__API_KEY=ollama
  LLM__MODEL_NAME=$SERVED_MODEL_NAME
EOF

  if [[ -n "$started_pid" ]]; then
    wait "$started_pid"
  fi
}

ensure_python_and_uv() {
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN="python3"
    else
      die "Python is required. Install Python 3.12 or pass --python."
    fi
  fi
  command -v uv >/dev/null 2>&1 || die "uv is required. Install it from https://docs.astral.sh/uv/"
}

install_vllm_linux() {
  ensure_python_and_uv

  if [[ -z "$PORT" ]]; then
    PORT="1234"
  fi

  if [[ ! -x "$VLLM_VENV_DIR/bin/python" ]]; then
    log "creating vLLM venv: $VLLM_VENV_DIR"
    "$PYTHON_BIN" -m venv "$VLLM_VENV_DIR"
  fi

  local need_install=false
  if [[ ! -x "$VLLM_VENV_DIR/bin/vllm" ]]; then
    need_install=true
  elif ! "$VLLM_VENV_DIR/bin/python" - "$VLLM_VERSION" <<'PY'
import importlib.metadata
import sys

expected = sys.argv[1]
try:
    installed = importlib.metadata.version("vllm")
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(1)

installed_base = installed.split("+", 1)[0]
raise SystemExit(0 if installed_base == expected else 1)
PY
  then
    need_install=true
  fi

  if [[ "$need_install" == true ]]; then
    log "installing vLLM $VLLM_VERSION for Linux $DEVICE"
    if [[ "$DEVICE" == "cpu" && -n "${VLLM_CPU_WHEEL_URL:-}" ]]; then
      uv pip install --python "$VLLM_VENV_DIR/bin/python" "$VLLM_CPU_WHEEL_URL" --torch-backend cpu
    elif [[ "$DEVICE" == "cpu" ]]; then
      uv pip install --python "$VLLM_VENV_DIR/bin/python" "vllm==$VLLM_VERSION" --torch-backend cpu
    else
      uv pip install --python "$VLLM_VENV_DIR/bin/python" "vllm==$VLLM_VERSION" --torch-backend auto
    fi
  fi
}

install_vllm_metal() {
  command -v curl >/dev/null 2>&1 || die "curl is required for vLLM-Metal installation"
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python 3.12 arm64 is required for vLLM-Metal"

  if [[ -z "$PORT" ]]; then
    PORT="1234"
  fi

  if [[ ! -x "$VLLM_METAL_VENV_DIR/bin/vllm" ]]; then
    log "installing vLLM-Metal into $VLLM_METAL_VENV_DIR"
    local tmp_installer
    tmp_installer="$(mktemp)"
    curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh -o "$tmp_installer"
    HOME="$ROOT_DIR" bash "$tmp_installer" "--$VLLM_METAL_CHANNEL"
    rm -f "$tmp_installer"
  fi

  if ! "$VLLM_METAL_VENV_DIR/bin/python" -c "import huggingface_hub" >/dev/null 2>&1; then
    "$VLLM_METAL_VENV_DIR/bin/python" -m pip install huggingface-hub
  fi
}

patch_mlx_processor_config() {
  local model_id="$1"
  "$VLLM_METAL_VENV_DIR/bin/python" - "$model_id" "$MODEL_CACHE_ROOT" <<'PY'
import json
import sys
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import snapshot_download

model_id, cache_root = sys.argv[1:3]
local_dir = Path(cache_root) / quote(model_id, safe="")
snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)

processor_config_path = local_dir / "processor_config.json"
preprocessor_config_path = local_dir / "preprocessor_config.json"

with processor_config_path.open() as f:
    processor_config = json.load(f)

image_processor = processor_config.get("image_processor")
if not isinstance(image_processor, dict):
    raise RuntimeError(f"{model_id} has no image_processor object in processor_config.json")

if image_processor.get("image_processor_type") == "Qwen3VLImageProcessor":
    image_processor["image_processor_type"] = "Qwen2VLImageProcessor"

processor_config["image_processor"] = image_processor
with processor_config_path.open("w") as f:
    json.dump(processor_config, f, indent=2)
    f.write("\n")

with preprocessor_config_path.open("w") as f:
    json.dump(image_processor, f, indent=2)
    f.write("\n")

print(local_dir)
PY
}

run_vllm() {
  local vllm_bin
  local default_max_model_len
  local default_gpu_memory
  local serve_args

  if [[ "$DEVICE" == "mps" ]]; then
    install_vllm_metal
    vllm_bin="$ROOT_DIR/$VLLM_METAL_VENV_DIR/bin/vllm"
    MODEL="${MODEL:-numind/NuExtract3-mlx-mxfp8}"
    MODEL="$(patch_mlx_processor_config "$MODEL")"
    default_gpu_memory="0.65"
  else
    install_vllm_linux
    vllm_bin="$ROOT_DIR/$VLLM_VENV_DIR/bin/vllm"
    if [[ "$DEVICE" == "gpu" ]]; then
      MODEL="${MODEL:-numind/NuExtract3-FP8}"
    else
      MODEL="${MODEL:-numind/NuExtract3-W8A8}"
    fi
    default_gpu_memory="0.90"
  fi

  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$default_gpu_memory}"

  serve_args=(
    serve "$MODEL"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --trust-remote-code
    --chat-template-content-format openai
    --limit-mm-per-prompt '{"image": 6, "video": 0}'
    --generation-config vllm
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  )

  if [[ "$DEVICE" == "mps" ]]; then
    serve_args+=(--max-num-seqs 1 --max-num-batched-tokens "$MAX_MODEL_LEN")
  else
    serve_args+=(--dtype auto --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}")
  fi

  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    serve_args+=("${EXTRA_ARGS[@]}")
  fi

  log "starting vLLM endpoint at http://$HOST:$PORT/v1"
  log "model: $MODEL, served name: $SERVED_MODEL_NAME"
  exec "$vllm_bin" "${serve_args[@]}"
}

validate_platform

if [[ "$CLEAN" == true ]]; then
  if [[ "$OS_NAME" == "Darwin" && "$DEVICE" == "cpu" ]]; then
    log "selected cpu on macOS -> using Ollama; no venv to remove"
    exit 0
  fi
  if [[ "$DEVICE" == "mps" ]]; then
    rm -rf "$VLLM_METAL_VENV_DIR"
    log "removed venv: $VLLM_METAL_VENV_DIR"
    exit 0
  fi
  rm -rf "$VLLM_VENV_DIR"
  log "removed venv: $VLLM_VENV_DIR"
  exit 0
fi

if [[ "$OS_NAME" == "Darwin" && "$DEVICE" == "cpu" ]]; then
  log "selected cpu on macOS -> using Ollama"
  run_ollama
else
  if [[ "$DEVICE" == "mps" ]]; then
    log "selected mps -> using vLLM-Metal"
  elif [[ "$DEVICE" == "gpu" ]]; then
    log "selected gpu on Linux -> using vLLM CUDA"
  else
    log "selected cpu on Linux -> using vLLM CPU"
  fi
  run_vllm
fi
