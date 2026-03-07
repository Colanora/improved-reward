#!/bin/bash
# SD3 counterfactual pseudo-pair mining on 2x H100 80GB.
# Usage:
#   bash scripts/single_node/cf_pairs_2h100.sh cope
#   bash scripts/single_node/cf_pairs_2h100.sh cope_lse config/counterfactual.py:sd3_cf_rerank_2gpu 8 0.25 logs/cf_pairs_sd3_cope_lse 29620 4

set -euo pipefail

SCORE_TYPE=${1:-cope}
CONFIG_NAME=${2:-config/counterfactual.py:sd3_cf_rerank_2gpu}
NUM_CANDIDATES=${3:-8}
MARGIN_THRESHOLD=${4:-0.25}
OUTPUT_DIR=${5:-logs/cf_pairs_sd3_${SCORE_TYPE}}
MAIN_PROCESS_PORT=${6:-29620}
NUM_NEGATIVES=${7:-}
NEGATIVE_MODE=${8:-auto}
MAX_PROMPTS=${9:-0}

case "${SCORE_TYPE}" in
  cope)
    if [ -z "${NUM_NEGATIVES}" ]; then
      NUM_NEGATIVES=1
    fi
    ;;
  cope_lse)
    if [ -z "${NUM_NEGATIVES}" ]; then
      NUM_NEGATIVES=4
    fi
    ;;
  raw|pmi)
    if [ -z "${NUM_NEGATIVES}" ]; then
      NUM_NEGATIVES=1
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
  scripts/build_cf_pairs_sd3.py \
  --config=${CONFIG_NAME} \
  --score_type=${SCORE_TYPE} \
  --num_candidates=${NUM_CANDIDATES} \
  --margin_threshold=${MARGIN_THRESHOLD} \
  --num_negatives=${NUM_NEGATIVES} \
  --negative_mode=${NEGATIVE_MODE} \
  --max_prompts=${MAX_PROMPTS} \
  --output_dir=${OUTPUT_DIR}
