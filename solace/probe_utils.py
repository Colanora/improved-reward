from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from solace.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt


def compute_text_embeddings_sd3(
    prompts: Sequence[str],
    text_encoders,
    tokenizers,
    device,
    max_sequence_length: int = 128,
    num_images_per_prompt: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders,
            tokenizers,
            prompt_list,
            max_sequence_length=max_sequence_length,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
    return prompt_embeds.to(device), pooled_prompt_embeds.to(device)


def repeat_condition_embeddings(
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prompt_embeds.shape[0] == batch_size:
        return prompt_embeds, pooled_prompt_embeds
    if prompt_embeds.shape[0] != 1 or pooled_prompt_embeds.shape[0] != 1:
        raise ValueError("Condition embeddings must already match the batch or be singleton.")
    return prompt_embeds.repeat(batch_size, 1, 1), pooled_prompt_embeds.repeat(batch_size, 1)


def build_antithetic_probes(
    k: int,
    shape: Sequence[int],
    device,
    dtype,
) -> torch.Tensor:
    if k <= 0 or k % 2 != 0:
        raise ValueError(f"k must be a positive even integer, got {k}.")
    half = torch.randn(k // 2, *shape, device=device, dtype=dtype)
    return torch.cat([half, -half], dim=0)


def make_probe_latents(
    x0: torch.Tensor,
    t_idx: torch.Tensor,
    eps: torch.Tensor,
    mode: str = "flow",
) -> torch.Tensor:
    if mode != "flow":
        raise NotImplementedError(f"Unsupported probe mode: {mode}")
    t = (t_idx.float() / 1000.0).view(-1, 1, 1, 1)
    return (1.0 - t).unsqueeze(0) * x0.unsqueeze(0) + t.unsqueeze(0) * eps


def select_probe_indices(
    timesteps: torch.Tensor,
    use_steps: Optional[Sequence[int] | int] = None,
    default_num_probe_steps: Optional[int] = None,
) -> List[int]:
    total_steps = timesteps.shape[1]
    if isinstance(use_steps, Iterable) and not isinstance(use_steps, (int, torch.Tensor)):
        return [int(index) for index in use_steps]

    if use_steps is None:
        use_steps = default_num_probe_steps if default_num_probe_steps is not None else total_steps

    count = max(1, min(int(use_steps), total_steps))
    start = total_steps - count
    return list(range(start, total_steps))


def _get_probe_settings(config) -> Dict[str, object]:
    cf_config = getattr(config, "cf", None)
    sds_config = getattr(getattr(config, "train", None), "sds", None)
    sample_config = getattr(config, "sample", None)

    k = getattr(cf_config, "k", getattr(sds_config, "k", 8))
    delta = getattr(cf_config, "delta", 1e-6)
    use_cfg_probe = bool(getattr(cf_config, "use_cfg_probe", False))
    normalize_per_step = bool(getattr(cf_config, "normalize_per_step", True))
    time_weighting = getattr(cf_config, "time_weighting", "mid")
    guidance_scale = float(getattr(sample_config, "guidance_scale", 1.0)) if sample_config is not None else 1.0
    probe_mode = getattr(cf_config, "probe_mode", "flow")
    default_steps = getattr(cf_config, "num_probe_steps", None)

    return {
        "k": int(k),
        "delta": float(delta),
        "use_cfg_probe": use_cfg_probe,
        "normalize_per_step": normalize_per_step,
        "time_weighting": time_weighting,
        "guidance_scale": guidance_scale,
        "probe_mode": probe_mode,
        "default_steps": default_steps,
    }


def _prepare_shared_probes(
    shared_eps: Optional[torch.Tensor],
    num_probe_steps: int,
    k: int,
    shape: Sequence[int],
    device,
    dtype,
) -> torch.Tensor:
    if shared_eps is None:
        return torch.stack(
            [build_antithetic_probes(k, shape, device=device, dtype=dtype) for _ in range(num_probe_steps)],
            dim=0,
        )

    if shared_eps.dim() == 5:
        return shared_eps.unsqueeze(0).repeat(num_probe_steps, 1, 1, 1, 1, 1)
    if shared_eps.dim() != 6:
        raise ValueError("shared_eps must have shape [K,B,C,H,W] or [T,K,B,C,H,W].")
    if shared_eps.shape[0] != num_probe_steps:
        raise ValueError("shared_eps first dimension must match the number of probe steps.")
    return shared_eps


def score_conditions_shared_probes(
    transformer,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    cond_list,
    shared_eps: Optional[torch.Tensor] = None,
    config=None,
    use_steps: Optional[Sequence[int] | int] = None,
    neg_prompt_embeds: Optional[torch.Tensor] = None,
    neg_pooled_prompt_embeds: Optional[torch.Tensor] = None,
):
    if not cond_list:
        raise ValueError("cond_list must contain at least one condition.")

    settings = _get_probe_settings(config)
    k = settings["k"]
    delta = settings["delta"]
    use_cfg_probe = settings["use_cfg_probe"]
    normalize_per_step = settings["normalize_per_step"]
    guidance_scale = settings["guidance_scale"]

    batch_size, channels, height, width = x0.shape
    device = x0.device
    transformer_dtype = next(transformer.parameters()).dtype
    step_indices = select_probe_indices(
        timesteps,
        use_steps=use_steps,
        default_num_probe_steps=settings["default_steps"],
    )
    probe_bank = _prepare_shared_probes(
        shared_eps=shared_eps,
        num_probe_steps=len(step_indices),
        k=k,
        shape=(batch_size, channels, height, width),
        device=device,
        dtype=x0.dtype,
    )

    if use_cfg_probe and (neg_prompt_embeds is None or neg_pooled_prompt_embeds is None):
        raise ValueError("CFG probing requires negative prompt embeddings.")

    raw_per_step_lists = {condition["name"]: [] for condition in cond_list}

    for step_position, timestep_index in enumerate(step_indices):
        t_idx = timesteps[:, timestep_index]
        eps = probe_bank[step_position]
        xt = make_probe_latents(x0=x0, t_idx=t_idx, eps=eps, mode=settings["probe_mode"])
        xt_flat = xt.reshape(k * batch_size, channels, height, width)
        xt_flat_model = xt_flat.to(transformer_dtype)
        t_flat = t_idx.repeat(k)
        eps_flat = eps.reshape(k * batch_size, channels, height, width)
        x0_flat = x0.unsqueeze(0).repeat(k, 1, 1, 1, 1).reshape(k * batch_size, channels, height, width)
        x0_flat_model = x0_flat.to(transformer_dtype)

        if use_cfg_probe:
            neg_pe = neg_prompt_embeds.repeat(k, 1, 1).to(transformer_dtype)
            neg_pp = neg_pooled_prompt_embeds.repeat(k, 1).to(transformer_dtype)

        for condition in cond_list:
            prompt_embeds = condition["prompt_embeds"]
            pooled_prompt_embeds = condition["pooled_prompt_embeds"]
            if prompt_embeds.shape[0] != batch_size or pooled_prompt_embeds.shape[0] != batch_size:
                raise ValueError("All condition embeddings must match the x0 batch size.")

            cond_pe = prompt_embeds.repeat(k, 1, 1).to(transformer_dtype)
            cond_pp = pooled_prompt_embeds.repeat(k, 1).to(transformer_dtype)

            if use_cfg_probe:
                v_uncond = transformer(
                    hidden_states=xt_flat_model,
                    timestep=t_flat,
                    encoder_hidden_states=neg_pe,
                    pooled_projections=neg_pp,
                    return_dict=False,
                )[0]
                v_cond = transformer(
                    hidden_states=xt_flat_model,
                    timestep=t_flat,
                    encoder_hidden_states=cond_pe,
                    pooled_projections=cond_pp,
                    return_dict=False,
                )[0]
                v_pred_flat = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_pred_flat = transformer(
                    hidden_states=xt_flat_model,
                    timestep=t_flat,
                    encoder_hidden_states=cond_pe,
                    pooled_projections=cond_pp,
                    return_dict=False,
                )[0]

            eps_hat_flat = v_pred_flat + x0_flat_model
            mse_flat = torch.mean((eps_hat_flat.float() - eps_flat.float()) ** 2, dim=(1, 2, 3))
            mse = mse_flat.view(k, batch_size).mean(dim=0)
            raw_step = -torch.log(mse + delta)
            raw_per_step_lists[condition["name"]].append(raw_step)

    raw_per_step = {
        name: torch.stack(values, dim=1)
        for name, values in raw_per_step_lists.items()
    }

    selected_t = timesteps[:, step_indices].float() / 1000.0
    if settings["time_weighting"] == "mid":
        weights = selected_t[0] * (1.0 - selected_t[0])
    else:
        weights = torch.ones(len(step_indices), device=device, dtype=x0.dtype)
    if torch.sum(weights).item() <= 0:
        weights = torch.ones_like(weights)

    normalized_per_step = {}
    scores = {}
    for name, step_scores in raw_per_step.items():
        if normalize_per_step and step_scores.shape[0] > 1:
            mean_t = step_scores.mean(dim=0, keepdim=True)
            std_t = step_scores.std(dim=0, keepdim=True).clamp_min(1e-6)
            normalized = (step_scores - mean_t) / std_t
        else:
            normalized = step_scores

        normalized_per_step[name] = normalized
        scores[name] = (normalized * weights.unsqueeze(0)).sum(dim=1) / weights.sum()

    return {
        "scores": scores,
        "normalized_per_step": normalized_per_step,
        "raw_per_step": raw_per_step,
        "weights": weights,
        "step_indices": step_indices,
        "shared_eps": probe_bank,
    }


def score_condition(
    transformer,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    neg_prompt_embeds: Optional[torch.Tensor],
    neg_pooled_prompt_embeds: Optional[torch.Tensor],
    config,
    use_steps: Optional[Sequence[int] | int] = None,
    shared_eps: Optional[torch.Tensor] = None,
):
    result = score_conditions_shared_probes(
        transformer=transformer,
        x0=x0,
        timesteps=timesteps,
        cond_list=[
            {
                "name": "condition",
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
            }
        ],
        shared_eps=shared_eps,
        config=config,
        use_steps=use_steps,
        neg_prompt_embeds=neg_prompt_embeds,
        neg_pooled_prompt_embeds=neg_pooled_prompt_embeds,
    )
    return (
        result["scores"]["condition"],
        result["normalized_per_step"]["condition"],
        result["raw_per_step"]["condition"],
    )
