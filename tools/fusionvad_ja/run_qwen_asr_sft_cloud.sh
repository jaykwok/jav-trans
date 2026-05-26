#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  if [[ "${CREATE_VENV:-0}" == "1" ]]; then
    python3 -m venv .venv
  else
    echo "Missing .venv/bin/python. Create it first, or rerun with CREATE_VENV=1." >&2
    exit 2
  fi
fi

HF_CACHE_DIR="${HF_CACHE_DIR:-$PROJECT_ROOT/datasets/hf-cache}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT_VALUE:-${HF_ENDPOINT:-}}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-$PROJECT_ROOT/datasets/train/qwen3-asr-ja-galgame/v1-pilot-asr200k}"
TRAIN_FILE="${TRAIN_FILE:-$SFT_OUTPUT_ROOT/qwen-sft/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-$SFT_OUTPUT_ROOT/qwen-sft/val.jsonl}"

DEFAULT_MODEL_DIR="$PROJECT_ROOT/models/Qwen-Qwen3-ASR-1.7B"
if [[ -f "$DEFAULT_MODEL_DIR/config.json" ]]; then
  QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-$DEFAULT_MODEL_DIR}"
else
  QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-Qwen/Qwen3-ASR-1.7B}"
fi

QWEN_SFT_SCRIPT_URL="${QWEN_SFT_SCRIPT_URL:-https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/finetuning/qwen3_asr_sft.py}"
QWEN_SFT_SCRIPT="${QWEN_SFT_SCRIPT:-$PROJECT_ROOT/agents/temp/qwen3_asr_sft.py}"
QWEN_SFT_OUTPUT_DIR="${QWEN_SFT_OUTPUT_DIR:-$PROJECT_ROOT/datasets/train/qwen3-asr-ja-galgame/v1-full-qwen3-asr-1.7b-sft}"
QWEN_SFT_LOG="${QWEN_SFT_LOG:-$PROJECT_ROOT/agents/temp/qwen3-asr-sft-cloud.run.log}"

BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-32}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-1}"
LOG_STEPS="${LOG_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

mkdir -p "$HF_CACHE_DIR" "$(dirname "$QWEN_SFT_SCRIPT")" "$QWEN_SFT_OUTPUT_DIR" "$(dirname "$QWEN_SFT_LOG")"

export HF_HOME="$HF_CACHE_DIR"
export HF_XET_HIGH_PERFORMANCE
if [[ -n "$HF_ENDPOINT_VALUE" ]]; then
  export HF_ENDPOINT="$HF_ENDPOINT_VALUE"
fi

if [[ "${INSTALL_QWEN_ASR_DEPS:-0}" == "1" ]]; then
  .venv/bin/pip install -U qwen-asr datasets librosa transformers
  if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
    MAX_JOBS="${MAX_JOBS:-4}" .venv/bin/pip install -U flash-attn --no-build-isolation
  fi
fi

if [[ ! -f "$QWEN_SFT_SCRIPT" ]]; then
  if [[ "${AUTO_DOWNLOAD_QWEN_SFT_SCRIPT:-1}" == "1" ]]; then
    curl -L "$QWEN_SFT_SCRIPT_URL" -o "$QWEN_SFT_SCRIPT"
  else
    echo "Missing Qwen SFT script: $QWEN_SFT_SCRIPT" >&2
    echo "Set AUTO_DOWNLOAD_QWEN_SFT_SCRIPT=1 or QWEN_SFT_SCRIPT=/path/to/qwen3_asr_sft.py." >&2
    exit 2
  fi
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Missing train file: $TRAIN_FILE" >&2
  echo "Run tools/fusionvad_ja/prepare_qwen_asr_cloud_assets.sh first." >&2
  exit 2
fi

COMMON_ARGS=(
  "$QWEN_SFT_SCRIPT"
  --model_path "$QWEN_MODEL_PATH"
  --train_file "$TRAIN_FILE"
  --output_dir "$QWEN_SFT_OUTPUT_DIR"
  --batch_size "$BATCH_SIZE"
  --grad_acc "$GRAD_ACC"
  --lr "$LR"
  --epochs "$EPOCHS"
  --log_steps "$LOG_STEPS"
  --save_strategy steps
  --save_steps "$SAVE_STEPS"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --num_workers "$NUM_WORKERS"
  --pin_memory "$PIN_MEMORY"
  --persistent_workers "$PERSISTENT_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
)

if [[ -n "$EVAL_FILE" && -f "$EVAL_FILE" ]]; then
  COMMON_ARGS+=(--eval_file "$EVAL_FILE")
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
  COMMON_ARGS+=(--resume_from "$RESUME_FROM")
elif [[ "${RESUME:-0}" == "1" ]]; then
  COMMON_ARGS+=(--resume 1)
fi

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  CMD=(.venv/bin/python -m torch.distributed.run --nproc_per_node "$NPROC_PER_NODE" "${COMMON_ARGS[@]}")
else
  CMD=(.venv/bin/python "${COMMON_ARGS[@]}")
fi

echo "HF_HOME=$HF_HOME"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  echo "HF_ENDPOINT=$HF_ENDPOINT"
fi
echo "QWEN_MODEL_PATH=$QWEN_MODEL_PATH"
echo "TRAIN_FILE=$TRAIN_FILE"
if [[ -n "$EVAL_FILE" && -f "$EVAL_FILE" ]]; then
  echo "EVAL_FILE=$EVAL_FILE"
fi
echo "QWEN_SFT_OUTPUT_DIR=$QWEN_SFT_OUTPUT_DIR"
echo "QWEN_SFT_LOG=$QWEN_SFT_LOG"
printf 'COMMAND='
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${CMD[@]}" 2>&1 | tee "$QWEN_SFT_LOG"
