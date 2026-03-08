#!/bin/bash
# Unified COPE vs SOLACE test on 2x H100 80GB.
# This runs one fp16 OCR rerank pass and compares all scoring rules from the same candidate pool:
#   - raw      (SOLACE-style baseline)
#   - pmi
#   - cope
#   - cope_lse
#
# Usage:
#   bash scripts/single_node/cf_vs_solace_fp16_2h100.sh
#   CF_SEED=43 bash scripts/single_node/cf_vs_solace_fp16_2h100.sh config/counterfactual.py:sd3_cf_rerank_2gpu_fp16 logs/cf_vs_solace_sd3_fp16 29640 8 0 4
#   bash scripts/single_node/cf_vs_solace_fp16_2h100.sh config/counterfactual.py:sd3_cf_structured_rerank_2gpu_fp16 logs/cf_vs_solace_structured_fp16 29641 8 0 4

set -euo pipefail

CONFIG_NAME=${1:-config/counterfactual.py:sd3_cf_rerank_2gpu_fp16}
OUTPUT_DIR=${2:-logs/cf_vs_solace_sd3_fp16}
MAIN_PROCESS_PORT=${3:-29640}
NUM_CANDIDATES=${4:-8}
MAX_PROMPTS=${5:-0}
NUM_NEGATIVES=${6:-4}

bash scripts/single_node/cf_rerank_fp16_2h100.sh \
  cope_lse \
  "${CONFIG_NAME}" \
  "${NUM_CANDIDATES}" \
  "${OUTPUT_DIR}" \
  "${MAIN_PROCESS_PORT}" \
  "${NUM_NEGATIVES}" \
  auto \
  "${MAX_PROMPTS}"

LATEST_RUN=$(
  find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
)

if [ -z "${LATEST_RUN}" ]; then
  echo "Could not locate the latest run directory under ${OUTPUT_DIR}" >&2
  exit 1
fi

python scripts/eval_counterfactual_sd3.py --input "${LATEST_RUN}" | tee "${LATEST_RUN}/compare_summary.json"

echo
echo "Unified comparison written to:"
echo "  ${LATEST_RUN}/summary.json"
echo "  ${LATEST_RUN}/compare_summary.json"
echo "TensorBoard:"
echo "  tensorboard --logdir ${OUTPUT_DIR}"
