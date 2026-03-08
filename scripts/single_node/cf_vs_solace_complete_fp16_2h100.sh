#!/bin/bash
# Complete COPE vs SOLACE comparison pipeline.
# Runs OCR and structured unified comparisons across one or more seeds and aggregates the reports.
#
# Usage:
#   bash scripts/single_node/cf_vs_solace_complete_fp16_2h100.sh
#   CF_SEEDS="42 43 44" bash scripts/single_node/cf_vs_solace_complete_fp16_2h100.sh logs/cf_vs_solace_complete_fp16 8 0 4

set -euo pipefail

OUTPUT_ROOT=${1:-logs/cf_vs_solace_complete_fp16}
NUM_CANDIDATES=${2:-8}
MAX_PROMPTS=${3:-0}
NUM_NEGATIVES=${4:-4}
SEEDS=${CF_SEEDS:-42}

OCR_RUNS=()
STRUCTURED_RUNS=()
OCR_PORT_BASE=29640
STRUCTURED_PORT_BASE=29740

run_index=0
for seed in ${SEEDS}; do
  OCR_OUTPUT_DIR="${OUTPUT_ROOT}/ocr_seed${seed}"
  STRUCTURED_OUTPUT_DIR="${OUTPUT_ROOT}/structured_seed${seed}"

  CF_SEED="${seed}" bash scripts/single_node/cf_vs_solace_fp16_2h100.sh \
    config/counterfactual.py:sd3_cf_rerank_2gpu_fp16 \
    "${OCR_OUTPUT_DIR}" \
    "$((OCR_PORT_BASE + run_index))" \
    "${NUM_CANDIDATES}" \
    "${MAX_PROMPTS}" \
    "${NUM_NEGATIVES}"

  OCR_RUNS+=("$(find "${OCR_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)")

  CF_SEED="${seed}" bash scripts/single_node/cf_vs_solace_fp16_2h100.sh \
    config/counterfactual.py:sd3_cf_structured_rerank_2gpu_fp16 \
    "${STRUCTURED_OUTPUT_DIR}" \
    "$((STRUCTURED_PORT_BASE + run_index))" \
    "${NUM_CANDIDATES}" \
    "${MAX_PROMPTS}" \
    "${NUM_NEGATIVES}"

  STRUCTURED_RUNS+=("$(find "${STRUCTURED_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)")
  run_index=$((run_index + 1))
done

mkdir -p "${OUTPUT_ROOT}"

OCR_ARGS=()
for run_dir in "${OCR_RUNS[@]}"; do
  OCR_ARGS+=(--input "${run_dir}")
done
python scripts/aggregate_counterfactual_runs.py "${OCR_ARGS[@]}" | tee "${OUTPUT_ROOT}/ocr_aggregate.json"

STRUCTURED_ARGS=()
for run_dir in "${STRUCTURED_RUNS[@]}"; do
  STRUCTURED_ARGS+=(--input "${run_dir}")
done
python scripts/aggregate_counterfactual_runs.py "${STRUCTURED_ARGS[@]}" | tee "${OUTPUT_ROOT}/structured_aggregate.json"

echo
echo "Aggregated reports written to:"
echo "  ${OUTPUT_ROOT}/ocr_aggregate.json"
echo "  ${OUTPUT_ROOT}/structured_aggregate.json"
