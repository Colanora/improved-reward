import datetime
import json
import os

from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from ml_collections import config_flags
import torch

import solace.rewards
from solace.counterfactual_sd3_utils import (
    encode_repeated_prompt,
    generate_candidates,
    json_ready_scores,
    load_prompt_dataset,
    load_sd3_pipeline,
    save_image_tensor,
    score_prompt_candidates,
    to_jsonable,
    write_jsonl_row,
)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/counterfactual.py:sd3_cf_rerank_2gpu", "Counterfactual reranking config.")
flags.DEFINE_string("dataset", None, "Optional dataset root or prompt file override.")
flags.DEFINE_string("dataset_split", None, "Optional split override.")
flags.DEFINE_integer("num_candidates", None, "Optional candidate-count override.")
flags.DEFINE_string("score_type", None, "Optional score type override: raw, pmi, cope, or cope_lse.")
flags.DEFINE_string("negative_mode", None, "Optional negative mode override: auto, ocr, count, spatial, attribute, unconditional.")
flags.DEFINE_integer("num_negatives", None, "Optional negative-count override.")
flags.DEFINE_integer("max_prompts", None, "Optional prompt limit override.")
flags.DEFINE_integer("seed_stride", None, "Optional per-prompt seed stride override.")
flags.DEFINE_string("output_dir", None, "Optional output directory override.")
logger = get_logger(__name__)


def _selected_index(scores: torch.Tensor) -> int:
    return int(torch.argmax(scores).item())


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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
    run_dir = os.path.join(config.output_dir, run_id)
    prompt_dir_root = os.path.join(run_dir, "prompts")
    _ensure_dir(prompt_dir_root)
    rank_results_path = os.path.join(run_dir, f"results_rank{accelerator.process_index}.jsonl")

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

    metric_fn = None
    if getattr(config.cf, "metrics", None):
        score_dict = {metric_name: 1.0 for metric_name in config.cf.metrics}
        metric_fn = getattr(solace.rewards, "multi_score")(device, score_dict)

    for prompt_index, prompt in local_prompts:
        prompt_output_dir = os.path.join(prompt_dir_root, f"{prompt_index:05d}")
        _ensure_dir(prompt_output_dir)

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
            images, latents, x0, timesteps = generate_candidates(
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

        score_map = score_result["scores"]
        selected = {
            "single": 0,
            "raw": _selected_index(score_map["raw"]),
            "pmi": _selected_index(score_map["pmi"]) if "pmi" in score_map else 0,
            "cope": _selected_index(score_map["cope"]) if "cope" in score_map else 0,
            "cope_lse": _selected_index(score_map["cope_lse"]) if "cope_lse" in score_map else 0,
        }
        selected["primary"] = selected.get(config.cf.score_type, selected["single"])

        candidate_image_paths = []
        candidate_latent_paths = []
        for candidate_index, seed in enumerate(seeds):
            image_path = os.path.join(prompt_output_dir, f"candidate_{candidate_index:02d}_seed{seed}.png")
            latent_path = os.path.join(prompt_output_dir, f"candidate_{candidate_index:02d}_seed{seed}.pt")
            candidate_image_paths.append(image_path)
            candidate_latent_paths.append(latent_path if config.cf.save_latents else None)

            if getattr(config.cf, "save_all_candidates", True):
                save_image_tensor(images[candidate_index], image_path, resize_to=config.resolution)
            if getattr(config.cf, "save_latents", False):
                torch.save({"x0": x0[candidate_index].detach().cpu()}, latent_path)

        for method_name, candidate_index in selected.items():
            save_image_tensor(
                images[candidate_index],
                os.path.join(prompt_output_dir, f"selected_{method_name}.png"),
                resize_to=config.resolution,
            )

        metric_values = {}
        if metric_fn is not None:
            metric_scores, _ = metric_fn(images, [prompt] * len(images), [{} for _ in range(len(images))], only_strict=False)
            metric_values = {name: to_jsonable(values) for name, values in metric_scores.items()}

        selected_metrics = {}
        for method_name, candidate_index in selected.items():
            selected_metrics[method_name] = {
                metric_name: float(metric_values[metric_name][candidate_index])
                for metric_name in metric_values
            }

        row = {
            "prompt_index": prompt_index,
            "prompt": prompt,
            "candidate_seeds": seeds,
            "negative_mode": score_result["negative_mode"],
            "negative_prompts": score_result["negative_prompts"],
            "scores": json_ready_scores(score_map),
            "selected_index": selected,
            "metrics": metric_values,
            "selected_metrics": selected_metrics,
            "candidate_image_paths": candidate_image_paths,
            "candidate_latent_paths": candidate_latent_paths,
        }
        write_jsonl_row(rank_results_path, row)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        merged_path = os.path.join(run_dir, "results.jsonl")
        with open(merged_path, "w", encoding="utf-8") as merged_handle:
            for rank in range(accelerator.num_processes):
                source_path = os.path.join(run_dir, f"results_rank{rank}.jsonl")
                if not os.path.exists(source_path):
                    continue
                with open(source_path, "r", encoding="utf-8") as source_handle:
                    for line in source_handle:
                        merged_handle.write(line)

        metadata = {
            "dataset": config.dataset,
            "dataset_split": config.cf.dataset_split,
            "num_candidates": config.cf.num_candidates,
            "num_probe_steps": config.cf.num_probe_steps,
            "k": config.cf.k,
            "score_type": config.cf.score_type,
            "negative_mode": config.cf.negative_mode,
            "num_negatives": config.cf.num_negatives,
            "shared_probes": config.cf.shared_probes,
        }
        with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    app.run(main)
