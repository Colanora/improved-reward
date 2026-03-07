#!/bin/bash
set -euo pipefail

bash scripts/single_node/cf_rerank_2h100.sh raw "${1:-config/counterfactual.py:sd3_cf_rerank_2gpu}" "${2:-8}" "${3:-logs/cf_rerank_sd3_raw}" "${4:-29610}" "${5:-1}" "${6:-auto}" "${7:-0}"
