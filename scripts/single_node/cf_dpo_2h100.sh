#!/bin/bash
# SD3 counterfactual LoRA-DPO on 2x H100 80GB.
# Usage:
#   bash scripts/single_node/cf_dpo_2h100.sh /path/to/pairs.jsonl
#   bash scripts/single_node/cf_dpo_2h100.sh /path/to/train_pairs.jsonl config/counterfactual.py:sd3_cf_dpo_2gpu logs/cf_dpo_sd3 29630
#   bash scripts/single_node/cf_dpo_2h100.sh /path/to/train_pairs.jsonl /path/to/val_pairs.jsonl config/counterfactual.py:sd3_cf_dpo_2gpu logs/cf_dpo_sd3 29630

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/single_node/cf_dpo_2h100.sh <pairs_jsonl> [val_pairs_jsonl] [config_name] [output_dir] [main_process_port]" >&2
  exit 1
fi

PAIRS_JSONL=$1
shift

VAL_PAIRS_JSONL=""
if [ $# -gt 0 ] && [[ "$1" == *.jsonl ]]; then
  VAL_PAIRS_JSONL=$1
  shift
fi

CONFIG_NAME=${1:-config/counterfactual.py:sd3_cf_dpo_2gpu}
OUTPUT_DIR=${2:-logs/cf_dpo_sd3}
MAIN_PROCESS_PORT=${3:-29630}

CMD=(
  accelerate launch
  --config_file scripts/accelerate_configs/two_h100_80g.yaml
  --num_processes=2
  --main_process_port=${MAIN_PROCESS_PORT}
  scripts/train_sd3_cf_dpo.py
  --config=${CONFIG_NAME}
  --pairs_jsonl=${PAIRS_JSONL}
  --output_dir=${OUTPUT_DIR}
)

if [ -n "${VAL_PAIRS_JSONL}" ]; then
  CMD+=(--val_pairs_jsonl=${VAL_PAIRS_JSONL})
fi

"${CMD[@]}"
