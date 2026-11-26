#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -d ".venv" ]]; then
  uv venv
fi
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

HF_CACHE_DIR="${HF_CACHE_DIR:-$PROJECT_ROOT/datasets/hf-cache}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT_VALUE:-${HF_ENDPOINT:-}}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-ASR-1.7B-hf}"
QWEN_MODEL_REVISION="${QWEN_MODEL_REVISION:-}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$PROJECT_ROOT/models/Qwen-Qwen3-ASR-1.7B-hf}"
SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-$PROJECT_ROOT/datasets/train/qwen3-asr-ja-galgame/v1-pilot-asr200k}"
SFT_MODE="${SFT_MODE:-full}"
SFT_REVISION="${SFT_REVISION:-}"
SFT_HF_AUDIO_FORMAT="${SFT_HF_AUDIO_FORMAT:-ogg}"
SFT_INCLUDE_SER="${SFT_INCLUDE_SER:-0}"
SFT_ASR_TRAIN_LIMIT="${SFT_ASR_TRAIN_LIMIT:-200000}"
SFT_ASR_VAL_LIMIT="${SFT_ASR_VAL_LIMIT:-1000}"
SFT_ASR_TEST_LIMIT="${SFT_ASR_TEST_LIMIT:-1000}"
SFT_SER_TRAIN_LIMIT="${SFT_SER_TRAIN_LIMIT:-50000}"
SFT_SER_VAL_LIMIT="${SFT_SER_VAL_LIMIT:-500}"
SFT_SER_TEST_LIMIT="${SFT_SER_TEST_LIMIT:-500}"

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
echo "SFT_HF_AUDIO_FORMAT=$SFT_HF_AUDIO_FORMAT"
echo "SFT_ASR_TRAIN_LIMIT=$SFT_ASR_TRAIN_LIMIT"
echo "SFT_INCLUDE_SER=$SFT_INCLUDE_SER"
if [[ "$SFT_INCLUDE_SER" == "1" ]]; then
  echo "SFT_SER_TRAIN_LIMIT=$SFT_SER_TRAIN_LIMIT"
fi

MODEL_ARGS=(
  "$QWEN_MODEL_ID"
  --local-dir "$QWEN_MODEL_DIR"
  --cache-dir "$HF_CACHE_DIR"
  --max-workers "${HF_MAX_WORKERS:-8}"
)
if [[ -n "$QWEN_MODEL_REVISION" ]]; then
  MODEL_ARGS+=(--revision "$QWEN_MODEL_REVISION")
fi

uv run --no-sync huggingface-cli download "${MODEL_ARGS[@]}"

SFT_ARGS=(
  tools/sft/prepare_qwen_asr_sft_dataset.py
  --mode "$SFT_MODE"
  --output-root "$SFT_OUTPUT_ROOT"
  --hf-cache-dir "$HF_CACHE_DIR"
  --hf-audio-format "$SFT_HF_AUDIO_FORMAT"
  --asr-train-limit "$SFT_ASR_TRAIN_LIMIT"
  --asr-val-limit "$SFT_ASR_VAL_LIMIT"
  --asr-test-limit "$SFT_ASR_TEST_LIMIT"
)
if [[ "$SFT_INCLUDE_SER" == "1" ]]; then
  SFT_ARGS+=(
    --ser-train-limit "$SFT_SER_TRAIN_LIMIT"
    --ser-val-limit "$SFT_SER_VAL_LIMIT"
    --ser-test-limit "$SFT_SER_TEST_LIMIT"
  )
else
  SFT_ARGS+=(--no-ser)
fi
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  SFT_ARGS+=(--hf-endpoint "$HF_ENDPOINT")
fi
if [[ -n "$SFT_REVISION" ]]; then
  SFT_ARGS+=(--revision "$SFT_REVISION")
fi

uv run --no-sync python "${SFT_ARGS[@]}"
