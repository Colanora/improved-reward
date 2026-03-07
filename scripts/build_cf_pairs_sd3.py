import datetime
import os

from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from ml_collections import config_flags
import torch

from solace.counterfactual_sd3_utils import (
    encode_repeated_prompt,
    generate_candidates,
    load_prompt_dataset,
    load_sd3_pipeline,
    save_image_tensor,
    score_prompt_candidates,
    write_jsonl_row,
)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/counterfactual.py:sd3_cf_rerank_2gpu", "Counterfactual pair-mining config.")
flags.DEFINE_string("dataset", None, "Optional dataset root or prompt file override.")
flags.DEFINE_string("dataset_split", None, "Optional split override.")
flags.DEFINE_integer("num_candidates", None, "Optional candidate-count override.")
flags.DEFINE_float("margin_threshold", 0.25, "Minimum winner-loser score gap.")
flags.DEFINE_string("score_type", None, "Optional scoring rule override.")
flags.DEFINE_string("negative_mode", None, "Optional negative mode override: auto, ocr, count, spatial, attribute, unconditional.")
flags.DEFINE_integer("num_negatives", None, "Optional negative-count override.")
flags.DEFINE_integer("max_prompts", None, "Optional prompt limit override.")
flags.DEFINE_integer("seed_stride", None, "Optional per-prompt seed stride override.")
flags.DEFINE_string("output_dir", None, "Optional output directory override.")
logger = get_logger(__name__)


def main(_):
    config = FLAGS.config
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    device = accelerator.device

    if FLAGS.dataset is not None:
        config.dataset = FLAGS.dataset
    if FLAGS.dataset_split is not None:
        config.cf.dataset_split = FLAGS.dataset_split
    if FLAGS.num_candidates is not None:
        config.cf.num_candidates = FLAGS.num_candidates
    if FLAGS.score_type is not None:
        config.cf.score_type = FLAGS.score_type
    if FLAGS.negative_mode is not None:
        config.cf.negative_mode = FLAGS.negative_mode
    if FLAGS.num_negatives is not None:
        config.cf.num_negatives = FLAGS.num_negatives
    if FLAGS.max_prompts is not None:
        config.cf.max_prompts = FLAGS.max_prompts
    if FLAGS.seed_stride is not None:
        config.cf.seed_stride = FLAGS.seed_stride
    if FLAGS.output_dir is not None:
        config.output_dir = FLAGS.output_dir

    run_id_holder = [datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if accelerator.is_main_process else None]
    if accelerator.num_processes > 1:
        torch.distributed.broadcast_object_list(run_id_holder, src=0)
    run_id = run_id_holder[0]
    run_dir = os.path.join(config.output_dir, f"pairs_{run_id}")
    pair_dir_root = os.path.join(run_dir, "pairs")
    os.makedirs(pair_dir_root, exist_ok=True)
    rank_pairs_path = os.path.join(run_dir, f"pairs_rank{accelerator.process_index}.jsonl")

    set_seed(config.seed, device_specific=True)
    logger.info("%s", config)

    pipeline, text_encoders, tokenizers = load_sd3_pipeline(
        config=config,
        device=device,
        is_local_main_process=accelerator.is_local_main_process,
    )

    prompts = load_prompt_dataset(
        config.dataset,
        split=config.cf.dataset_split,
        max_prompts=config.cf.max_prompts,
    )
    local_prompts = prompts[accelerator.process_index::accelerator.num_processes]

    for prompt_index, prompt in local_prompts:
        prompt_pair_dir = os.path.join(pair_dir_root, f"{prompt_index:05d}")
        os.makedirs(prompt_pair_dir, exist_ok=True)

        seeds = [
            int(config.seed + prompt_index * config.cf.seed_stride + candidate_index)
            for candidate_index in range(config.cf.num_candidates)
        ]

        prompt_embeds, pooled_prompt_embeds = encode_repeated_prompt(
            prompt,
            batch_size=config.cf.num_candidates,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            device=device,
        )
        negative_prompt_embeds, negative_pooled_prompt_embeds = encode_repeated_prompt(
            "",
            batch_size=config.cf.num_candidates,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            device=device,
        )

        with torch.no_grad():
            images, _, x0, timesteps = generate_candidates(
                pipeline=pipeline,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                config=config,
                seeds=seeds,
            )
            score_result = score_prompt_candidates(
                transformer=pipeline.transformer,
                prompt=prompt,
                x0=x0,
                timesteps=timesteps,
                text_encoders=text_encoders,
                tokenizers=tokenizers,
                config=config,
                device=device,
            )

        score_values = score_result["scores"][config.cf.score_type]
        winner_index = int(torch.argmax(score_values).item())
        loser_index = int(torch.argmin(score_values).item())
        winner_score = float(score_values[winner_index].item())
        loser_score = float(score_values[loser_index].item())
        if winner_score - loser_score < FLAGS.margin_threshold:
            continue

        winner_seed = seeds[winner_index]
        loser_seed = seeds[loser_index]
        winner_path = os.path.join(prompt_pair_dir, f"winner_seed{winner_seed}.png")
        loser_path = os.path.join(prompt_pair_dir, f"loser_seed{loser_seed}.png")
        winner_latent_path = os.path.join(prompt_pair_dir, f"winner_seed{winner_seed}.pt")
        loser_latent_path = os.path.join(prompt_pair_dir, f"loser_seed{loser_seed}.pt")

        save_image_tensor(images[winner_index], winner_path, resize_to=config.resolution)
        save_image_tensor(images[loser_index], loser_path, resize_to=config.resolution)
        torch.save({"x0": x0[winner_index].detach().cpu()}, winner_latent_path)
        torch.save({"x0": x0[loser_index].detach().cpu()}, loser_latent_path)

        row = {
            "prompt_index": prompt_index,
            "prompt": prompt,
            "winner_seed": winner_seed,
            "loser_seed": loser_seed,
            "winner_score": winner_score,
            "loser_score": loser_score,
            "winner_path": winner_path,
            "loser_path": loser_path,
            "winner_latent_path": winner_latent_path,
            "loser_latent_path": loser_latent_path,
            "negative_prompt": score_result["negative_prompts"][0] if score_result["negative_prompts"] else "",
            "negative_prompts": score_result["negative_prompts"],
            "negative_mode": score_result["negative_mode"],
            "score_type": config.cf.score_type,
        }
        write_jsonl_row(rank_pairs_path, row)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merged_path = os.path.join(run_dir, "pairs.jsonl")
        with open(merged_path, "w", encoding="utf-8") as merged_handle:
            for rank in range(accelerator.num_processes):
                source_path = os.path.join(run_dir, f"pairs_rank{rank}.jsonl")
                if not os.path.exists(source_path):
                    continue
                with open(source_path, "r", encoding="utf-8") as source_handle:
                    for line in source_handle:
                        merged_handle.write(line)


if __name__ == "__main__":
    app.run(main)
