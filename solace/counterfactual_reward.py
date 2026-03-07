from typing import Dict, List, Optional, Sequence

import torch

from solace.probe_utils import score_conditions_shared_probes


def _condition(name: str, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "name": name,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }


def compute_counterfactual_scores(
    transformer,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    positive_prompt_embeds: torch.Tensor,
    positive_pooled_prompt_embeds: torch.Tensor,
    config,
    unconditional_prompt_embeds: Optional[torch.Tensor] = None,
    unconditional_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds_list: Optional[Sequence[torch.Tensor]] = None,
    negative_pooled_prompt_embeds_list: Optional[Sequence[torch.Tensor]] = None,
    use_steps: Optional[Sequence[int] | int] = None,
    shared_eps: Optional[torch.Tensor] = None,
    probe_neg_prompt_embeds: Optional[torch.Tensor] = None,
    probe_neg_pooled_prompt_embeds: Optional[torch.Tensor] = None,
):
    cond_list = [
        _condition("positive", positive_prompt_embeds, positive_pooled_prompt_embeds),
    ]

    if unconditional_prompt_embeds is not None and unconditional_pooled_prompt_embeds is not None:
        cond_list.append(
            _condition("unconditional", unconditional_prompt_embeds, unconditional_pooled_prompt_embeds)
        )

    negative_names: List[str] = []
    if negative_prompt_embeds_list is not None and negative_pooled_prompt_embeds_list is not None:
        for index, (neg_pe, neg_pp) in enumerate(zip(negative_prompt_embeds_list, negative_pooled_prompt_embeds_list)):
            name = f"negative_{index}"
            negative_names.append(name)
            cond_list.append(_condition(name, neg_pe, neg_pp))

    probe_result = score_conditions_shared_probes(
        transformer=transformer,
        x0=x0,
        timesteps=timesteps,
        cond_list=cond_list,
        shared_eps=shared_eps,
        config=config,
        use_steps=use_steps,
        neg_prompt_embeds=probe_neg_prompt_embeds,
        neg_pooled_prompt_embeds=probe_neg_pooled_prompt_embeds,
    )

    score_map = dict(probe_result["scores"])
    score_map["raw"] = score_map["positive"]

    if "unconditional" in score_map:
        score_map["pmi"] = score_map["positive"] - score_map["unconditional"]

    if negative_names:
        first_negative = score_map[negative_names[0]]
        score_map["cope"] = score_map["positive"] - first_negative
        neg_stack = torch.stack([score_map[name] for name in negative_names], dim=0)
        score_map["cope_lse"] = score_map["positive"] - torch.logsumexp(neg_stack, dim=0)

    return {
        "scores": score_map,
        "normalized_per_step": probe_result["normalized_per_step"],
        "raw_per_step": probe_result["raw_per_step"],
        "weights": probe_result["weights"],
        "step_indices": probe_result["step_indices"],
        "shared_eps": probe_result["shared_eps"],
        "negative_names": negative_names,
    }


def raw_confidence(
    transformer,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    config,
    **kwargs,
) -> torch.Tensor:
    result = compute_counterfactual_scores(
        transformer=transformer,
        x0=x0,
        timesteps=timesteps,
        positive_prompt_embeds=prompt_embeds,
        positive_pooled_prompt_embeds=pooled_prompt_embeds,
        config=config,
        **kwargs,
    )
    return result["scores"]["raw"]


def pmi_confidence(
    transformer,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    unconditional_prompt_embeds: torch.Tensor,
    unconditional_pooled_prompt_embeds: torch.Tensor,
    config,
    **kwargs,
) -> torch.Tensor:
    result = compute_counterfactual_scores(
        transformer=transformer,
        x0=x0,
        timesteps=timesteps,
        positive_prompt_embeds=prompt_embeds,
        positive_pooled_prompt_embeds=pooled_prompt_embeds,
        unconditional_prompt_embeds=unconditional_prompt_embeds,
        unconditional_pooled_prompt_embeds=unconditional_pooled_prompt_embeds,
        config=config,
        **kwargs,
    )
    return result["scores"]["pmi"]


def counterfactual_confidence(
    transformer,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_pooled_prompt_embeds: torch.Tensor,
    config,
    **kwargs,
) -> torch.Tensor:
    result = compute_counterfactual_scores(
        transformer=transformer,
        x0=x0,
        timesteps=timesteps,
        positive_prompt_embeds=prompt_embeds,
        positive_pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds_list=[negative_prompt_embeds],
        negative_pooled_prompt_embeds_list=[negative_pooled_prompt_embeds],
        config=config,
        **kwargs,
    )
    return result["scores"]["cope"]


def counterfactual_confidence_lse(
    transformer,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_prompt_embeds_list: Sequence[torch.Tensor],
    negative_pooled_prompt_embeds_list: Sequence[torch.Tensor],
    config,
    **kwargs,
) -> torch.Tensor:
    result = compute_counterfactual_scores(
        transformer=transformer,
        x0=x0,
        timesteps=timesteps,
        positive_prompt_embeds=prompt_embeds,
        positive_pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds_list=negative_prompt_embeds_list,
        negative_pooled_prompt_embeds_list=negative_pooled_prompt_embeds_list,
        config=config,
        **kwargs,
    )
    return result["scores"]["cope_lse"]
