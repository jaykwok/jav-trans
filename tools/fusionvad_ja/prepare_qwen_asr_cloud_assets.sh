#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

HF_CACHE_DIR="${HF_CACHE_DIR:-$PROJECT_ROOT/datasets/hf-cache}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT_VALUE:-${HF_ENDPOINT:-}}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-ASR-1.7B}"
QWEN_MODEL_REVISION="${QWEN_MODEL_REVISION:-}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$PROJECT_ROOT/models/Qwen-Qwen3-ASR-1.7B}"
SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-$PROJECT_ROOT/datasets/train/qwen3-asr-ja-galgame/v1-full}"
SFT_MODE="${SFT_MODE:-full}"
SFT_REVISION="${SFT_REVISION:-}"

mkdir -p "$HF_CACHE_DIR" "$QWEN_MODEL_DIR" "$(dirname "$SFT_OUTPUT_ROOT")"

export HF_HOME="$HF_CACHE_DIR"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
if [[ -n "$HF_ENDPOINT_VALUE" ]]; then
  export HF_ENDPOINT="$HF_ENDPOINT_VALUE"
fi

echo "HF_HOME=$HF_HOME"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  echo "HF_ENDPOINT=$HF_ENDPOINT"
fi
echo "QWEN_MODEL_ID=$QWEN_MODEL_ID"
if [[ -n "$QWEN_MODEL_REVISION" ]]; then
  echo "QWEN_MODEL_REVISION=$QWEN_MODEL_REVISION"
fi
echo "QWEN_MODEL_DIR=$QWEN_MODEL_DIR"
echo "SFT_OUTPUT_ROOT=$SFT_OUTPUT_ROOT"

MODEL_ARGS=(
  "$QWEN_MODEL_ID"
  --local-dir "$QWEN_MODEL_DIR"
  --cache-dir "$HF_CACHE_DIR"
  --max-workers "${HF_MAX_WORKERS:-8}"
)
if [[ -n "$QWEN_MODEL_REVISION" ]]; then
  MODEL_ARGS+=(--revision "$QWEN_MODEL_REVISION")
fi

.venv/bin/huggingface-cli download "${MODEL_ARGS[@]}"

SFT_ARGS=(
  tools/fusionvad_ja/prepare_qwen_asr_sft_dataset.py
  --mode "$SFT_MODE"
  --output-root "$SFT_OUTPUT_ROOT"
  --hf-cache-dir "$HF_CACHE_DIR"
)
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  SFT_ARGS+=(--hf-endpoint "$HF_ENDPOINT")
fi
if [[ -n "$SFT_REVISION" ]]; then
  SFT_ARGS+=(--revision "$SFT_REVISION")
fi

.venv/bin/python "${SFT_ARGS[@]}"
