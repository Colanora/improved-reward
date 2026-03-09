#!/bin/bash
# Launch matched 1-GPU GRPO pilots for raw SOLACE vs COPE-LSE.
# Usage:
#   bash scripts/single_node/grpo_sd3_raw_vs_cope_lse_2x1gpu.sh
#   bash scripts/single_node/grpo_sd3_raw_vs_cope_lse_2x1gpu.sh logs/grpo_compare_sd3_1gpu 42 config/solace.py:general_ocr_sd3_grpo_compare_1gpu_base

set -euo pipefail

OUTPUT_ROOT=${1:-logs/grpo_compare_sd3_1gpu}
SEED=${2:-42}
CONFIG_NAME=${3:-config/solace.py:general_ocr_sd3_grpo_compare_1gpu_base}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d_%H%M%S)}
GROUP_ROOT="${OUTPUT_ROOT}/${RUN_GROUP}"

cleanup() {
  if [ -n "${RAW_PID:-}" ] && kill -0 "${RAW_PID}" 2>/dev/null; then
    kill "${RAW_PID}" 2>/dev/null || true
  fi
  if [ -n "${COPE_PID:-}" ] && kill -0 "${COPE_PID}" 2>/dev/null; then
    kill "${COPE_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

mkdir -p "${GROUP_ROOT}"

bash scripts/single_node/grpo_sd3_raw_1gpu.sh \
  "${CONFIG_NAME}" \
  "${GROUP_ROOT}" \
  "raw" \
  "${SEED}" \
  0 &
RAW_PID=$!

bash scripts/single_node/grpo_sd3_cope_lse_1gpu.sh \
  "${CONFIG_NAME}" \
  "${GROUP_ROOT}" \
  "cope_lse" \
  "${SEED}" \
  1 &
COPE_PID=$!

raw_status=0
cope_status=0

while true; do
  if ! kill -0 "${RAW_PID}" 2>/dev/null; then
    wait "${RAW_PID}" || raw_status=$?
    if [ "${raw_status}" -ne 0 ]; then
      kill "${COPE_PID}" 2>/dev/null || true
      wait "${COPE_PID}" || true
      exit "${raw_status}"
    fi
    wait "${COPE_PID}" || cope_status=$?
    break
  fi

  if ! kill -0 "${COPE_PID}" 2>/dev/null; then
    wait "${COPE_PID}" || cope_status=$?
    if [ "${cope_status}" -ne 0 ]; then
      kill "${RAW_PID}" 2>/dev/null || true
      wait "${RAW_PID}" || true
      exit "${cope_status}"
    fi
    wait "${RAW_PID}" || raw_status=$?
    break
  fi

  sleep 5
done

trap - EXIT INT TERM

if [ "${raw_status}" -ne 0 ] || [ "${cope_status}" -ne 0 ]; then
  exit 1
fi

echo
echo "Raw and COPE-LSE pilot runs finished."
echo "TensorBoard:"
echo "  tensorboard --logdir ${GROUP_ROOT}"
