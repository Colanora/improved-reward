import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from solace.baseline_prompts import build_counterfactuals, resolve_counterfactual_mode
from solace.counterfactual_reward import compute_counterfactual_scores
from solace.probe_utils import (
    compute_text_embeddings_sd3,
    repeat_condition_embeddings,
)


def get_inference_dtype(mixed_precision: str):
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_prompt_dataset(dataset_root: str, split: str = "test", max_prompts: int = 0) -> List[Tuple[int, str]]:
    dataset_path = dataset_root
    if os.path.isdir(dataset_path):
        dataset_path = os.path.join(dataset_path, f"{split}.txt")

    with open(dataset_path, "r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]

    if max_prompts and max_prompts > 0:
        prompts = prompts[:max_prompts]
    return list(enumerate(prompts))


def build_generators(device, seeds: Sequence[int]) -> List[torch.Generator]:
    return [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]


def load_sd3_pipeline(config, device, is_local_main_process: bool = True):
    from diffusers import StableDiffusion3Pipeline

    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = get_inference_dtype(config.mixed_precision)
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(device, dtype=inference_dtype)
    pipeline.transformer.to(device, dtype=inference_dtype)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    return pipeline, text_encoders, tokenizers


def encode_repeated_prompt(
    prompt: str,
    batch_size: int,
    text_encoders,
    tokenizers,
    device,
    max_sequence_length: int = 128,
):
    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings_sd3(
        [prompt],
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        device=device,
        max_sequence_length=max_sequence_length,
    )
    return repeat_condition_embeddings(prompt_embeds, pooled_prompt_embeds, batch_size=batch_size)


def generate_candidates(
    pipeline,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_pooled_prompt_embeds: torch.Tensor,
    config,
    seeds: Sequence[int],
):
    from solace.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob

    generators = build_generators(prompt_embeds.device, seeds)
    with torch.no_grad():
        images, latents, _ = pipeline_with_logprob(
            pipeline,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=config.sample.eval_num_steps,
            guidance_scale=config.sample.guidance_scale,
            output_type="pt",
            height=config.resolution,
            width=config.resolution,
            noise_level=0.0,
            generator=generators,
        )

    stacked_latents = torch.stack(latents, dim=1)
    x0 = stacked_latents[:, -1]
    timesteps = pipeline.scheduler.timesteps.repeat(len(seeds), 1).to(prompt_embeds.device)
    return images, stacked_latents, x0, timesteps


def score_prompt_candidates(
    transformer,
    prompt: str,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    text_encoders,
    tokenizers,
    config,
    device,
    negative_prompts_override: Sequence[str] | None = None,
    negative_mode_override: str | None = None,
):
    batch_size = x0.shape[0]
    positive_prompt_embeds, positive_pooled_prompt_embeds = encode_repeated_prompt(
        prompt,
        batch_size=batch_size,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        device=device,
    )
    unconditional_prompt_embeds, unconditional_pooled_prompt_embeds = encode_repeated_prompt(
        "",
        batch_size=batch_size,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        device=device,
    )
    if negative_prompts_override is None:
        negative_prompts = build_counterfactuals(
            prompt,
            mode=config.cf.negative_mode,
            n_neg=config.cf.num_negatives,
        )
    else:
        negative_prompts = list(negative_prompts_override)
    negative_mode = (
        negative_mode_override
        if negative_mode_override is not None
        else resolve_counterfactual_mode(prompt, mode=config.cf.negative_mode)
    )

    negative_prompt_embeds_list = []
    negative_pooled_prompt_embeds_list = []
    for negative_prompt in negative_prompts:
        neg_pe, neg_pp = encode_repeated_prompt(
            negative_prompt,
            batch_size=batch_size,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            device=device,
        )
        negative_prompt_embeds_list.append(neg_pe)
        negative_pooled_prompt_embeds_list.append(neg_pp)

    if getattr(config.cf, "shared_probes", True):
        score_result = compute_counterfactual_scores(
            transformer=transformer,
            x0=x0,
            timesteps=timesteps,
            positive_prompt_embeds=positive_prompt_embeds,
            positive_pooled_prompt_embeds=positive_pooled_prompt_embeds,
            unconditional_prompt_embeds=unconditional_prompt_embeds,
            unconditional_pooled_prompt_embeds=unconditional_pooled_prompt_embeds,
            negative_prompt_embeds_list=negative_prompt_embeds_list,
            negative_pooled_prompt_embeds_list=negative_pooled_prompt_embeds_list,
            config=config,
            use_steps=config.cf.num_probe_steps,
            probe_neg_prompt_embeds=unconditional_prompt_embeds,
            probe_neg_pooled_prompt_embeds=unconditional_pooled_prompt_embeds,
        )
    else:
        score_result = compute_counterfactual_scores(
            transformer=transformer,
            x0=x0,
            timesteps=timesteps,
            positive_prompt_embeds=positive_prompt_embeds,
            positive_pooled_prompt_embeds=positive_pooled_prompt_embeds,
            config=config,
            use_steps=config.cf.num_probe_steps,
            probe_neg_prompt_embeds=unconditional_prompt_embeds,
            probe_neg_pooled_prompt_embeds=unconditional_pooled_prompt_embeds,
        )
        pmi_result = compute_counterfactual_scores(
            transformer=transformer,
            x0=x0,
            timesteps=timesteps,
            positive_prompt_embeds=positive_prompt_embeds,
            positive_pooled_prompt_embeds=positive_pooled_prompt_embeds,
            unconditional_prompt_embeds=unconditional_prompt_embeds,
            unconditional_pooled_prompt_embeds=unconditional_pooled_prompt_embeds,
            config=config,
            use_steps=config.cf.num_probe_steps,
            probe_neg_prompt_embeds=unconditional_prompt_embeds,
            probe_neg_pooled_prompt_embeds=unconditional_pooled_prompt_embeds,
        )
        score_result["scores"]["unconditional"] = pmi_result["scores"]["unconditional"]
        score_result["scores"]["pmi"] = pmi_result["scores"]["pmi"]

        negative_scores = []
        for index, (neg_pe, neg_pp) in enumerate(zip(negative_prompt_embeds_list, negative_pooled_prompt_embeds_list)):
            neg_result = compute_counterfactual_scores(
                transformer=transformer,
                x0=x0,
                timesteps=timesteps,
                positive_prompt_embeds=positive_prompt_embeds,
                positive_pooled_prompt_embeds=positive_pooled_prompt_embeds,
                negative_prompt_embeds_list=[neg_pe],
                negative_pooled_prompt_embeds_list=[neg_pp],
                config=config,
                use_steps=config.cf.num_probe_steps,
                probe_neg_prompt_embeds=unconditional_prompt_embeds,
                probe_neg_pooled_prompt_embeds=unconditional_pooled_prompt_embeds,
            )
            name = f"negative_{index}"
            singleton_score = neg_result["scores"]["negative_0"]
            score_result["scores"][name] = singleton_score
            negative_scores.append(singleton_score)

        if negative_scores:
            score_result["scores"]["cope"] = score_result["scores"]["positive"] - negative_scores[0]
            negative_stack = torch.stack(negative_scores, dim=0)
            score_result["scores"]["cope_lse"] = score_result["scores"]["positive"] - torch.logsumexp(negative_stack, dim=0)

    score_result["negative_prompts"] = negative_prompts
    score_result["negative_mode"] = negative_mode
    return score_result


def save_image_tensor(image_tensor: torch.Tensor, output_path: str, resize_to: int | None = None):
    image = image_tensor.detach().cpu().clamp(0, 1)
    pil = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype("uint8"))
    if resize_to is not None:
        pil = pil.resize((resize_to, resize_to))
    pil.save(output_path)


def json_ready_scores(score_map: Dict[str, torch.Tensor]) -> Dict[str, List[float]]:
    return {name: tensor.detach().cpu().tolist() for name, tensor in score_map.items()}


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_jsonl_row(path: str, row: Dict[str, object]):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
