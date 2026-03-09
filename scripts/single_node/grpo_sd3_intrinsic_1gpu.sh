#!/bin/bash
# SD3.5-Medium intrinsic GRPO training on a single H100.
# Usage:
#   bash scripts/single_node/grpo_sd3_intrinsic_1gpu.sh raw
#   bash scripts/single_node/grpo_sd3_intrinsic_1gpu.sh cope_lse config/solace.py:general_ocr_sd3_grpo_compare_1gpu_pilot_base logs/grpo_compare_sd3_1gpu raw_run 42 0

set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PADDLE_PDX_MODEL_SOURCE="${PADDLE_PDX_MODEL_SOURCE:-huggingface}"
export PADDLE_PDX_HUGGING_FACE_ENDPOINT="${PADDLE_PDX_HUGGING_FACE_ENDPOINT:-$HF_ENDPOINT}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -n "${CONDA_PREFIX:-}" ]; then
  NVJITLINK_DIR="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cu13/lib"
  if [ -f "${NVJITLINK_DIR}/libnvJitLink.so.13" ]; then
    export LD_LIBRARY_PATH="${NVJITLINK_DIR}:${LD_LIBRARY_PATH:-}"
  fi
fi

SCORE_TYPE=${1:-raw}
CONFIG_NAME=${2:-config/solace.py:general_ocr_sd3_grpo_compare_1gpu_base}
LOG_ROOT=${3:-logs/grpo_compare_sd3_1gpu}
RUN_NAME=${4:-sd3_grpo_${SCORE_TYPE}_1gpu}
SEED=${5:-42}
GPU_ID=${6:-0}

case "${SCORE_TYPE}" in
  raw|pmi|cope)
    NUM_NEGATIVES=1
    ;;
  cope_lse)
    NUM_NEGATIVES=4
    ;;
  *)
    echo "Unsupported score type: ${SCORE_TYPE}" >&2
    exit 1
    ;;
esac

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch \
  --config_file scripts/accelerate_configs/one_h100_80g.yaml \
  --num_processes=1 \
  scripts/train_sd3_self.py \
  --config="${CONFIG_NAME}" \
  --config.cf.score_type="${SCORE_TYPE}" \
  --config.cf.num_negatives="${NUM_NEGATIVES}" \
  --config.logdir="${LOG_ROOT}" \
  --config.run_name="${RUN_NAME}" \
  --config.save_dir="${LOG_ROOT}/${RUN_NAME}" \
  --config.logging_backend=tensorboard \
  --config.seed="${SEED}"
