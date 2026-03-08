#!/bin/bash
# WAN 2.1 SOLACE Training (single node)
# Usage:
#   8 GPU:  bash scripts/single_node/grpo_wan_self.sh

set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PADDLE_PDX_MODEL_SOURCE="${PADDLE_PDX_MODEL_SOURCE:-huggingface}"
export PADDLE_PDX_HUGGING_FACE_ENDPOINT="${PADDLE_PDX_HUGGING_FACE_ENDPOINT:-$HF_ENDPOINT}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"

accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes=8 \
    --main_process_port 29503 \
    scripts/train_wan2_1_self.py \
    --config config/solace.py:general_ocr_wan2_1
