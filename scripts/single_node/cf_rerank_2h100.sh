#!/bin/bash
# SD3 counterfactual reranking on 2x H100 80GB.
# Usage:
#   bash scripts/single_node/cf_rerank_2h100.sh cope
#   bash scripts/single_node/cf_rerank_2h100.sh raw config/counterfactual.py:sd3_cf_rerank_2gpu 8 logs/cf_rerank_sd3_raw 29610
#   bash scripts/single_node/cf_rerank_2h100.sh cope_lse config/counterfactual.py:sd3_cf_structured_rerank_2gpu 8 logs/cf_structured_cope_lse 29610 4

set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PADDLE_PDX_MODEL_SOURCE="${PADDLE_PDX_MODEL_SOURCE:-huggingface}"
export PADDLE_PDX_HUGGING_FACE_ENDPOINT="${PADDLE_PDX_HUGGING_FACE_ENDPOINT:-$HF_ENDPOINT}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"

if [ -n "${CONDA_PREFIX:-}" ]; then
  NVJITLINK_DIR="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cu13/lib"
  if [ -f "${NVJITLINK_DIR}/libnvJitLink.so.13" ]; then
    export LD_LIBRARY_PATH="${NVJITLINK_DIR}:${LD_LIBRARY_PATH:-}"
  fi
fi

SCORE_TYPE=${1:-cope}
CONFIG_NAME=${2:-config/counterfactual.py:sd3_cf_rerank_2gpu}
NUM_CANDIDATES=${3:-8}
OUTPUT_DIR=${4:-logs/cf_rerank_sd3_${SCORE_TYPE}}
MAIN_PROCESS_PORT=${5:-29610}
NUM_NEGATIVES=${6:-}
NEGATIVE_MODE=${7:-auto}
MAX_PROMPTS=${8:-0}
DISABLE_METRICS=${CF_DISABLE_METRICS:-0}
SEED_OVERRIDE=${CF_SEED:-}
EXTRA_ARGS=()

if [ "${DISABLE_METRICS}" = "1" ]; then
  EXTRA_ARGS+=(--disable_metrics)
fi

if [ -n "${SEED_OVERRIDE}" ]; then
  EXTRA_ARGS+=(--seed="${SEED_OVERRIDE}")
fi

case "${SCORE_TYPE}" in
  raw|pmi|cope)
    if [ -z "${NUM_NEGATIVES}" ]; then
      NUM_NEGATIVES=1
    fi
    ;;
  cope_lse)
    if [ -z "${NUM_NEGATIVES}" ]; then
      NUM_NEGATIVES=4
    fi
    ;;
  *)
    echo "Unsupported score type: ${SCORE_TYPE}" >&2
    exit 1
    ;;
esac

accelerate launch \
  --config_file scripts/accelerate_configs/two_h100_80g.yaml \
  --num_processes=2 \
  --main_process_port=${MAIN_PROCESS_PORT} \
  scripts/rerank_counterfactual_sd3.py \
  --config=${CONFIG_NAME} \
  --score_type=${SCORE_TYPE} \
  --num_candidates=${NUM_CANDIDATES} \
  --num_negatives=${NUM_NEGATIVES} \
  --negative_mode=${NEGATIVE_MODE} \
  --max_prompts=${MAX_PROMPTS} \
  --output_dir=${OUTPUT_DIR} \
  "${EXTRA_ARGS[@]}"
