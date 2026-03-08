#!/bin/bash
# SD3.5 Self-Confidence Training
# 1 GPU:
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3_self.py --config config/solace.py:general_ocr_sd3_1gpu
# 8 GPU:
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PADDLE_PDX_MODEL_SOURCE="${PADDLE_PDX_MODEL_SOURCE:-huggingface}"
export PADDLE_PDX_HUGGING_FACE_ENDPOINT="${PADDLE_PDX_HUGGING_FACE_ENDPOINT:-$HF_ENDPOINT}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"

accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29501 scripts/train_sd3_self.py --config config/solace.py:general_ocr_sd3_8gpu
