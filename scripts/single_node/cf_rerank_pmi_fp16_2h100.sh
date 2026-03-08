#!/bin/bash
set -euo pipefail

bash scripts/single_node/cf_rerank_fp16_2h100.sh pmi "${1:-config/counterfactual.py:sd3_cf_rerank_2gpu_fp16}" "${2:-8}" "${3:-logs/cf_rerank_sd3_pmi_fp16}" "${4:-29611}" "${5:-1}" "${6:-unconditional}" "${7:-0}"
