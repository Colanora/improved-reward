# 1-GPU GRPO Compare Notes

This note records the current launch guidance for the post-training GRPO comparison between the original SOLACE raw reward and COPE-LSE on `2x H100 80GB`.

## Current Status

The full compare configs preserve the original SOLACE 1-GPU batch shape:

- `config/solace.py:general_ocr_sd3_grpo_raw_1gpu`
- `config/solace.py:general_ocr_sd3_grpo_cope_lse_1gpu`

These now enable activation checkpointing, TensorBoard logging, `TOKENIZERS_PARALLELISM=false`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Even with those fixes, the full `8 x 8` 1-GPU microbatch shape can still OOM on the current software stack.

## Recommended Fallback

Use the matched `fit80g` configs instead:

- `config/solace.py:general_ocr_sd3_grpo_raw_1gpu_fit80g`
- `config/solace.py:general_ocr_sd3_grpo_cope_lse_1gpu_fit80g`

These keep the comparison fair:

- same trainer and LoRA setup
- same OCR dataset and eval path
- same seed
- same samples per epoch: `64`
- same effective train batch size: `32`

What changes is only the per-step memory footprint:

- full config: `train_batch_size=8`, `num_image_per_prompt=8`, `num_batches_per_epoch=8`
- `fit80g`: `train_batch_size=4`, `num_image_per_prompt=4`, `num_batches_per_epoch=16`

`gradient_accumulation_steps` is increased accordingly so the optimization budget stays matched.

## Launch Commands

Separate terminals:

```bash
bash scripts/single_node/grpo_sd3_raw_1gpu_fit80g.sh \
  config/solace.py:general_ocr_sd3_grpo_raw_1gpu_fit80g \
  logs/grpo_compare_sd3_1gpu \
  raw_fit80g \
  42 \
  0
```

```bash
bash scripts/single_node/grpo_sd3_cope_lse_1gpu_fit80g.sh \
  config/solace.py:general_ocr_sd3_grpo_cope_lse_1gpu_fit80g \
  logs/grpo_compare_sd3_1gpu \
  cope_lse_fit80g \
  42 \
  1
```

Single launcher:

```bash
bash scripts/single_node/grpo_sd3_raw_vs_cope_lse_2x1gpu.sh \
  logs/grpo_compare_sd3_1gpu \
  42 \
  config/solace.py:general_ocr_sd3_grpo_compare_1gpu_fit80g_base
```

TensorBoard:

```bash
tensorboard --logdir logs/grpo_compare_sd3_1gpu
```

## Interpretation

Stage-1 reranking results in [COPE_vs_SOLACE_results.md](/home/wjy/scifi/SOLACE/docs/COPE_vs_SOLACE_results.md) are unchanged. This note only affects the stage-2 GRPO comparison path.
