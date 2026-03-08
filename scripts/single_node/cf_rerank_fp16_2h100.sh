#!/bin/bash
# SD3 counterfactual reranking on 2x H100 80GB using fp16.
# Usage:
#   bash scripts/single_node/cf_rerank_fp16_2h100.sh raw
#   bash scripts/single_node/cf_rerank_fp16_2h100.sh cope config/counterfactual.py:sd3_cf_rerank_2gpu_fp16 8 logs/cf_rerank_sd3_cope_fp16 29612
#   bash scripts/single_node/cf_rerank_fp16_2h100.sh cope config/counterfactual.py:sd3_cf_structured_rerank_2gpu_fp16 8 logs/cf_structured_cope_fp16 29612

set -euo pipefail

SCORE_TYPE=${1:-cope}
CONFIG_NAME=${2:-config/counterfactual.py:sd3_cf_rerank_2gpu_fp16}
NUM_CANDIDATES=${3:-8}
OUTPUT_DIR=${4:-logs/cf_rerank_sd3_${SCORE_TYPE}_fp16}
MAIN_PROCESS_PORT=${5:-29610}
NUM_NEGATIVES=${6:-}
NEGATIVE_MODE=${7:-auto}
MAX_PROMPTS=${8:-0}

bash scripts/single_node/cf_rerank_2h100.sh \
  "${SCORE_TYPE}" \
  "${CONFIG_NAME}" \
  "${NUM_CANDIDATES}" \
  "${OUTPUT_DIR}" \
  "${MAIN_PROCESS_PORT}" \
  "${NUM_NEGATIVES}" \
  "${NEGATIVE_MODE}" \
  "${MAX_PROMPTS}"
