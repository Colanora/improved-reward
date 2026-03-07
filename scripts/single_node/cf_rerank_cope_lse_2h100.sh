#!/bin/bash
set -euo pipefail

bash scripts/single_node/cf_rerank_2h100.sh cope_lse "${1:-config/counterfactual.py:sd3_cf_rerank_2gpu}" "${2:-8}" "${3:-logs/cf_rerank_sd3_cope_lse}" "${4:-29613}" "${5:-4}" "${6:-auto}" "${7:-0}"
