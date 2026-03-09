#!/bin/bash
set -euo pipefail

CONFIG_NAME=${1:-config/solace.py:general_ocr_sd3_grpo_cope_lse_1gpu}
LOG_ROOT=${2:-logs/grpo_compare_sd3_1gpu}
RUN_NAME=${3:-sd3_grpo_cope_lse_1gpu}
SEED=${4:-42}
GPU_ID=${5:-0}

bash scripts/single_node/grpo_sd3_intrinsic_1gpu.sh cope_lse "${CONFIG_NAME}" "${LOG_ROOT}" "${RUN_NAME}" "${SEED}" "${GPU_ID}"
