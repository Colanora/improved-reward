#!/bin/bash
set -euo pipefail

bash scripts/single_node/cf_pairs_2h100.sh cope_lse "${1:-config/counterfactual.py:sd3_cf_rerank_2gpu}" "${2:-8}" "${3:-0.25}" "${4:-logs/cf_pairs_sd3_cope_lse}" "${5:-29621}" "${6:-4}" "${7:-auto}" "${8:-0}"
