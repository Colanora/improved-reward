from collections import defaultdict
import contextlib
import os
import datetime
import inspect
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import solace.prompts
# add back external rewards
import solace.rewards
from solace.stat_tracking import PerPromptStatTracker
from solace.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from solace.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from solace.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from solace.baseline_prompts import build_counterfactuals
from solace.counterfactual_reward import compute_counterfactual_scores
import torch
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from solace.ema import EMAModuleWrapper

try:
    import wandb
except ImportError:
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
logger = get_logger(__name__)

def adapter_off_ctx(m):
    """
    Returns a context manager that disables LoRA adapters if present.
    Works for both DDP-wrapped and plain PEFT models.
    """
    base = getattr(m, "module", m)  # unwrap DDP if needed
    if hasattr(base, "disable_adapter"):          # PEFT PeftModel
        return base.disable_adapter()              # context manager
    return contextlib.nullcontext()                # no-op if not LoRA


class ExperimentLogger:
    def __init__(self, backend, run_dir):
        self.backend = backend
        self.run_dir = run_dir
        self.writer = None

        if backend == "tensorboard":
            if SummaryWriter is None:
                raise ImportError("TensorBoard logging requested but `tensorboard` is not installed.")
            self.writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))
        elif backend == "wandb":
            if wandb is None:
                raise ImportError("W&B logging requested but `wandb` is not installed.")
            wandb.init(project="solace", dir=run_dir)
        elif backend != "none":
            raise ValueError(f"Unsupported logging backend: {backend}")

    def log_scalars(self, metrics, step):
        if self.backend == "tensorboard":
            for key, value in metrics.items():
                if value is None:
                    continue
                self.writer.add_scalar(key, value, step)
        elif self.backend == "wandb":
            wandb.log(metrics, step=step)

    def log_images(self, tag, images, step, captions=None):
        if not images:
            return
        if self.backend == "tensorboard":
            image_tensor = torch.stack([image.detach().cpu().float().clamp(0, 1) for image in images], dim=0)
            self.writer.add_images(tag, image_tensor, step)
        elif self.backend == "wandb":
            captions = captions or [None] * len(images)
            wandb.log(
                {
                    tag: [
                        wandb.Image(
                            image.detach().cpu().permute(1, 2, 0).numpy(),
                            caption=caption,
                        )
                        for image, caption in zip(images, captions)
                    ]
                },
                step=step,
            )

    def close(self):
        if self.writer is not None:
            self.writer.close()


def get_run_dir(config):
    return os.path.join(config.logdir, config.run_name)


def get_grpo_probe_step_indices(num_train_timesteps: int):
    start = int(num_train_timesteps * 0.5)
    return list(range(start, num_train_timesteps))


def maybe_enable_gradient_checkpointing(module):
    if hasattr(module, "enable_gradient_checkpointing"):
        module.enable_gradient_checkpointing()
        return True
    if hasattr(module, "gradient_checkpointing_enable"):
        module.gradient_checkpointing_enable()
        return True
    return False

# --------------------- Datasets & Sampler ---------------------

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}
    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}
    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0
    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]
    def set_epoch(self, epoch):
        self.epoch = epoch

# --------------------- Helpers ---------------------

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def build_batched_negative_prompts(prompts, config):
    prompt_negative_lists = []
    negative_modes = []

    for prompt in prompts:
        negatives = build_counterfactuals(
            prompt,
            mode=getattr(config.cf, "negative_mode", "auto"),
            n_neg=max(1, int(getattr(config.cf, "num_negatives", 1))),
        )
        prompt_negative_lists.append(negatives)
        negative_modes.append(getattr(config.cf, "negative_mode", "auto"))

    return prompt_negative_lists, negative_modes


def compute_intrinsic_score_map(
    transformer,
    prompts,
    x0,
    timesteps,
    prompt_embeds,
    pooled_prompt_embeds,
    text_encoders,
    tokenizers,
    neg_prompt_embed,
    neg_pooled_prompt_embed,
    config,
    device,
    use_steps,
    autocast_ctx,
):
    batch_size = x0.shape[0]
    score_type = getattr(config.cf, "score_type", "raw")
    score_buffers = {}
    selected_per_step = None

    with torch.no_grad():
        if score_type == "raw":
            raw_score, raw_norm_per_step, _ = sds_self_confidence_scalar(
                transformer=transformer,
                x0=x0,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                neg_prompt_embed=neg_prompt_embed,
                neg_pooled_prompt_embed=neg_pooled_prompt_embed,
                config=config,
                device=device,
                autocast_ctx=autocast_ctx,
                use_steps=use_steps,
            )
            score_buffers["raw"] = raw_score
            selected_score = score_buffers["raw"]
            selected_per_step = raw_norm_per_step
            negative_modes = ["raw"] * batch_size
        elif score_type == "pmi":
            unconditional_prompt_embeds = neg_prompt_embed.repeat(batch_size, 1, 1)
            unconditional_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(batch_size, 1)
            score_result = compute_counterfactual_scores(
                transformer=transformer,
                x0=x0,
                timesteps=timesteps,
                positive_prompt_embeds=prompt_embeds,
                positive_pooled_prompt_embeds=pooled_prompt_embeds,
                unconditional_prompt_embeds=unconditional_prompt_embeds,
                unconditional_pooled_prompt_embeds=unconditional_pooled_prompt_embeds,
                config=config,
                use_steps=use_steps,
                probe_neg_prompt_embeds=unconditional_prompt_embeds,
                probe_neg_pooled_prompt_embeds=unconditional_pooled_prompt_embeds,
            )
            for key in ("raw", "pmi"):
                score_buffers[key] = score_result["scores"][key]
            selected_score = score_buffers["pmi"]
            selected_per_step = score_result["normalized_per_step"]["positive"]
            negative_modes = ["unconditional"] * batch_size
        else:
            prompt_negative_lists, negative_modes = build_batched_negative_prompts(prompts, config)
            grouped_indices = defaultdict(list)
            for index, negatives in enumerate(prompt_negative_lists):
                grouped_indices[len(negatives)].append(index)

            score_buffers = defaultdict(lambda: torch.empty(batch_size, device=device, dtype=torch.float32))
            for _, indices in grouped_indices.items():
                index_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
                subgroup_size = len(indices)
                unconditional_prompt_embeds = neg_prompt_embed.repeat(subgroup_size, 1, 1)
                unconditional_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(subgroup_size, 1)

                negative_prompt_embeds_list = []
                negative_pooled_prompt_embeds_list = []
                num_group_negatives = len(prompt_negative_lists[indices[0]])
                for slot_index in range(num_group_negatives):
                    slot_prompts = [prompt_negative_lists[index][slot_index] for index in indices]
                    neg_pe, neg_pp = compute_text_embeddings(
                        slot_prompts,
                        text_encoders,
                        tokenizers,
                        max_sequence_length=128,
                        device=device,
                    )
                    negative_prompt_embeds_list.append(neg_pe)
                    negative_pooled_prompt_embeds_list.append(neg_pp)

                score_result = compute_counterfactual_scores(
                    transformer=transformer,
                    x0=x0.index_select(0, index_tensor),
                    timesteps=timesteps.index_select(0, index_tensor),
                    positive_prompt_embeds=prompt_embeds.index_select(0, index_tensor),
                    positive_pooled_prompt_embeds=pooled_prompt_embeds.index_select(0, index_tensor),
                    unconditional_prompt_embeds=unconditional_prompt_embeds,
                    unconditional_pooled_prompt_embeds=unconditional_pooled_prompt_embeds,
                    negative_prompt_embeds_list=negative_prompt_embeds_list,
                    negative_pooled_prompt_embeds_list=negative_pooled_prompt_embeds_list,
                    config=config,
                    use_steps=use_steps,
                    probe_neg_prompt_embeds=unconditional_prompt_embeds,
                    probe_neg_pooled_prompt_embeds=unconditional_pooled_prompt_embeds,
                )

                for key in ("raw", "pmi", "cope", "cope_lse"):
                    if key in score_result["scores"]:
                        score_buffers[key][index_tensor] = score_result["scores"][key]

                subgroup_selected_per_step = score_result["normalized_per_step"]["positive"]
                if selected_per_step is None:
                    selected_per_step = torch.empty(
                        batch_size,
                        subgroup_selected_per_step.shape[1],
                        device=device,
                        dtype=subgroup_selected_per_step.dtype,
                    )
                selected_per_step[index_tensor] = subgroup_selected_per_step

            score_buffers = {key: value for key, value in score_buffers.items()}
            selected_score = score_buffers[score_type]

    return {
        "scores": score_buffers,
        "selected_score": selected_score,
        "selected_per_step": selected_per_step,
        "negative_modes": negative_modes,
    }

def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

# --------------------- PPO/GRPO Logprob Step ---------------------

def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )
    return prev_sample, log_prob, prev_sample_mean, std_dev_t

# --------------------- SDS ONLY: self-certainty probe ---------------------

def sds_self_confidence_scalar(
    transformer,
    x0,                        # [B,C,H,W] clean latent from trajectory
    timesteps,                 # [B, T_all] scheduler timesteps (ints, e.g., 999..)
    prompt_embeds,             # [B,S,D]
    pooled_prompt_embeds,      # [B,Dp]
    neg_prompt_embed,          # [1,S,D] (already computed once)
    neg_pooled_prompt_embed,   # [1,Dp]
    config,
    device,
    autocast_ctx,
    use_steps=None,            # int: number of training steps to use
):
    """
    Returns: sds_scalar [B] and (optionally) per-step scores [B,T_used] for debugging.
    """
    B, C, H, W = x0.shape
    K = getattr(config.train, "sds", {}).get("k", 8)
    step_stride = getattr(config.train, "sds", {}).get("use_step_stride", 1)
    scale = getattr(config.train, "sds", {}).get("scale", 1.0)

    # Determine how many steps to use
    T_all = timesteps.shape[1]
    T_used = min(use_steps if use_steps is not None else T_all, T_all)
    # allocate (exclude last clean index)
    T_used = min(T_used, T_all - 1)  # make sure we skip the final clean latent
    if T_used <= 0:
        return torch.zeros(B, device=device), None

    sds_per_step = torch.zeros(B, T_used, device=device, dtype=torch.float32)

    # CFG?
    use_cfg = False# bool(getattr(config.train, "cfg", False)) and (float(config.sample.guidance_scale) > 1.0)
    gs = float(config.sample.guidance_scale) if use_cfg else 1.0

    # Prebuild repeated embeds for probes
    cond_pe = prompt_embeds.repeat(K, 1, 1)          # [K*B,S,D] after tile of B below
    cond_pp = pooled_prompt_embeds.repeat(K, 1)      # [K*B,Dp]
    neg_pe  = neg_prompt_embed.repeat(K * B, 1, 1)   # [K*B,S,D]
    neg_pp  = neg_pooled_prompt_embed.repeat(K * B, 1)  # [K*B,Dp]

    with torch.no_grad():
        with autocast_ctx():
            for j in range(int (T_used * 0.5), T_used, step_stride):
                t_idx = timesteps[:, j]                       # [B]
                # normalize to [0,1] (SD3 scheduler uses 0..999 or similar)
                t = t_idx.float() / 1000.0
                t_expanded = t.view(B, 1, 1, 1)               # [B,1,1,1]

                # K probes
                eps = torch.randn(K // 2, B, C, H, W, device=device, dtype=x0.dtype)
                eps = torch.cat((eps, -eps), dim=0)
                
                # simple FM blend: x_t = (1-t) * x0 + t * eps   (same as in your previous SDS probe)
                xt = (1.0 - t_expanded) * x0.unsqueeze(0) + t_expanded * eps  # [K,B,C,H,W]
                xt_flat = xt.reshape(K * B, C, H, W)
                t_flat  = t_idx.repeat(K)

                if use_cfg:
                    v_u = transformer(
                        hidden_states=xt_flat,
                        timestep=t_flat,
                        encoder_hidden_states=neg_pe,
                        pooled_projections=neg_pp,
                        return_dict=False,
                    )[0]

                    v_c = transformer(
                        hidden_states=xt_flat,
                        timestep=t_flat,
                        encoder_hidden_states=cond_pe,
                        pooled_projections=cond_pp,
                        return_dict=False,
                    )[0]                                                # [2*K*B,C,H,W]

                    v_pred_flat = v_u + gs * (v_c - v_u)                # [K*B,C,H,W]
                else:
                    v_pred_flat = transformer(
                        hidden_states=xt_flat,
                        timestep=t_flat,
                        encoder_hidden_states=cond_pe,                  # [K*B,S,D] (will be broadcast over B)
                        pooled_projections=cond_pp,                     # [K*B,Dp]
                        return_dict=False,
                    )[0]                                                # [K*B,C,H,W]

                x0_flat = x0.unsqueeze(0).repeat(K, 1, 1, 1, 1).reshape(K * B, C, H, W)
                eps_hat_flat = v_pred_flat + x0_flat                    # FM: eps_hat = v + x0
                mse_flat = torch.mean((eps_hat_flat - eps.reshape(K * B, C, H, W)) ** 2, dim=(1, 2, 3))  # [K*B]
                mse = mse_flat.view(K, B).mean(dim=0)                   # [B]
                sds_step = -torch.log(mse + 1e-6)                       # higher is better
                sds_per_step[:, j] = sds_step

    # Per-step (across batch) normalization for stability
    mean_t = sds_per_step.mean(dim=0, keepdim=True)
    std_t  = sds_per_step.std(dim=0, keepdim=True).clamp_min(1e-6)
    sds_norm = (sds_per_step - mean_t) / std_t                          # [B,T_used]

    # Time weighting: emphasize mid-range steps
    float_t = (timesteps[:, :T_used].float() / 1000.0)
    sds_norm = sds_norm * (float_t * (1.0 - float_t))

    sds_scalar = scale * sds_norm.mean(dim=1)                           # [B]
    return sds_scalar, sds_norm, sds_per_step

# --------------------- (Optional) SDS Eval ---------------------

def eval_sds(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator,
             global_step, num_train_timesteps, ema, transformer_trainable_parameters, neg_prompt_embed, neg_pooled_prompt_embed, autocast):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    # simple SDS-eval: generate a batch and compute SDS scalars
    all_sds = []
    last_batch_images_gather = None
    last_batch_prompts_gather = None
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval (SDS): ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
        )
        with autocast():
            with torch.no_grad():
                images, latents, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=neg_prompt_embed.repeat(len(prompts), 1, 1),
                    negative_pooled_prompt_embeds=neg_pooled_prompt_embed.repeat(len(prompts), 1),
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                )
        latents = torch.stack(latents, dim=1)  # [B, T+1, C,H,W]
        x0 = latents[:, -1]
        timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(accelerator.device)

        sds_scalar, _, sds_per_step = sds_self_confidence_scalar(
            transformer=pipeline.transformer,
            x0=x0,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            neg_prompt_embed=neg_prompt_embed,
            neg_pooled_prompt_embed=neg_pooled_prompt_embed,
            config=config,
            device=accelerator.device,
            autocast_ctx=autocast,
            use_steps=num_train_timesteps,
        )
        sds_scalar_g = accelerator.gather(sds_scalar).cpu().numpy()
        all_sds.append(sds_scalar_g)

        last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
        last_batch_prompt_ids = tokenizers[0](
            prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
        ).input_ids.to(accelerator.device)
        last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
        last_batch_prompts_gather = pipeline.tokenizer.batch_decode(last_batch_prompt_ids_gather, skip_special_tokens=True)

    all_sds = np.concatenate(all_sds) if len(all_sds) > 0 else np.array([])
    if accelerator.is_main_process and last_batch_images_gather is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_sds = [all_sds[index] if index < len(all_sds) else 0.0 for index in sample_indices]
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | sds: {sds:.2f}",
                        )
                        for idx, (prompt, sds) in enumerate(zip(sampled_prompts, sampled_sds))
                    ],
                    "eval_reward_sds": float(all_sds[all_sds == all_sds].mean()) if all_sds.size > 0 else 0.0,
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator,
         global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters,
         experiment_logger):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)
    all_intrinsic = defaultdict(list)
    probe_steps = get_grpo_probe_step_indices(num_train_timesteps)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
        )
        if len(prompt_embeds) < len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, latents, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                )
        latents = torch.stack(latents, dim=1)
        x0 = latents[:, -1]
        timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(accelerator.device)
        intrinsic_result = compute_intrinsic_score_map(
            transformer=pipeline.transformer,
            prompts=prompts,
            x0=x0,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            neg_prompt_embed=neg_prompt_embed,
            neg_pooled_prompt_embed=neg_pooled_prompt_embed,
            config=config,
            device=accelerator.device,
            use_steps=num_train_timesteps if config.cf.score_type == "raw" else probe_steps,
            autocast_ctx=autocast,
        )
        for key, value in intrinsic_result["scores"].items():
            gathered_intrinsic = accelerator.gather(value).cpu().numpy()
            all_intrinsic[key].append(gathered_intrinsic)
        # external reward
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)

    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    all_intrinsic = {key: np.concatenate(value) for key, value in all_intrinsic.items()}
    if accelerator.is_main_process and experiment_logger is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            image_tensors = []
            captions = []
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                image_tensors.append(torch.tensor(image, dtype=torch.float32))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for prompt, reward in zip(sampled_prompts, sampled_rewards):
                captions.append(
                    f"{prompt:.1000} | "
                    + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10)
                )
            experiment_logger.log_images("eval_images", image_tensors, step=global_step, captions=captions)
            metrics = {f"eval_reward/{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()}
            metrics.update({f"eval_intrinsic/{key}": float(np.mean(value)) for key, value in all_intrinsic.items()})
            metrics["eval_intrinsic/selected"] = float(np.mean(all_intrinsic[config.cf.score_type]))
            experiment_logger.log_scalars(metrics, step=global_step)
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)


# --------------------- Main ---------------------

def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
    run_dir = get_run_dir(config)
    if not config.save_dir:
        config.save_dir = run_dir

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=run_dir,
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * max(1, num_train_timesteps),
    )
    experiment_logger = None
    if accelerator.is_main_process:
        experiment_logger = ExperimentLogger(config.logging_backend, run_dir)
    logger.info(f"\n{config}")
    if accelerator.is_main_process and config.logging_backend == "tensorboard":
        logger.info(f"TensorBoard logdir: {os.path.join(run_dir, 'tensorboard')}")

    set_seed(config.seed, device_specific=True)

    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device)

    if config.activation_checkpointing:
        enabled = maybe_enable_gradient_checkpointing(pipeline.transformer)
        if not enabled:
            logger.warning("activation_checkpointing=True but transformer does not expose a checkpointing API.")

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj","attn.add_q_proj","attn.add_v_proj","attn.to_add_out",
            "attn.to_k","attn.to_out.0","attn.to_q","attn.to_v",
        ]
        transformer_lora_config = LoraConfig(r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules)
        if config.train.lora_path:
            load_signature = inspect.signature(PeftModel.from_pretrained)
            if "is_trainable" in load_signature.parameters:
                pipeline.transformer = PeftModel.from_pretrained(
                    pipeline.transformer,
                    config.train.lora_path,
                    is_trainable=True,
                )
            else:
                pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
                for name, parameter in pipeline.transformer.named_parameters():
                    if "lora_" in name:
                        parameter.requires_grad = True
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Install bitsandbytes for 8-bit Adam: `pip install bitsandbytes`")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare external reward functions (evaluation-time)
    reward_fn = getattr(solace.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(solace.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    # executor for async reward computation during eval
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Datasets/loaders
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset  = TextPromptDataset(config.dataset, 'test')
        train_sampler = DistributedKRepeatSampler(train_dataset, config.sample.train_batch_size,
                                                  config.sample.num_image_per_prompt, accelerator.num_processes,
                                                  accelerator.process_index, seed=42)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=1,
                                      collate_fn=TextPromptDataset.collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=config.sample.test_batch_size,
                                     collate_fn=TextPromptDataset.collate_fn, shuffle=False, num_workers=8)
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset  = GenevalPromptDataset(config.dataset, 'test')
        train_sampler = DistributedKRepeatSampler(train_dataset, config.sample.train_batch_size,
                                                  config.sample.num_image_per_prompt, accelerator.num_processes,
                                                  accelerator.process_index, seed=42)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=1,
                                      collate_fn=GenevalPromptDataset.collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=config.sample.test_batch_size,
                                     collate_fn=GenevalPromptDataset.collate_fn, shuffle=False, num_workers=8)
    else:
        raise NotImplementedError("Only general_ocr and geneval are supported.")

    # Embeddings for CFG (neg)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds  = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds  = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # autocast choice
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        transformer, optimizer, train_dataloader, test_dataloader
    )

    # --- Intrinsic GRPO training ---
    samples_per_epoch = (config.sample.train_batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch)
    total_train_batch_size = (config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps)

    logger.info("***** Running training (Intrinsic GRPO) *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device  = {config.train.batch_size}")
    logger.info(f"  Grad Accum steps (x timesteps) = {config.train.gradient_accumulation_steps} x {num_train_timesteps}")
    logger.info(f"  Total samples/epoch = {samples_per_epoch}")
    logger.info(f"  GRPO updates/inner epoch = {samples_per_epoch // max(1,total_train_batch_size)}")
    logger.info(f"  Inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while epoch < config.num_epochs:
        # ========== EVAL ==========
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0:
            eval(
                pipeline, test_dataloader, text_encoders, tokenizers,
                config, accelerator, global_step,
                eval_reward_fn, executor, autocast,
                num_train_timesteps, ema, transformer_trainable_parameters,
                experiment_logger,
            )
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        # ========== SAMPLING + intrinsic scoring ==========
        pipeline.transformer.eval()
        samples = []
        prompts_last = None
        images_last = None
        for i in tqdm(range(config.sample.num_batches_per_epoch),
                      desc=f"Epoch {epoch}: sampling", disable=not accelerator.is_local_main_process, position=0):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt",
            ).input_ids.to(accelerator.device)

            generator = create_generator(prompts, base_seed=epoch*10000+i) if config.sample.same_latent else None
            with autocast():
                with torch.no_grad():
                    images, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator
                    )

            latents = torch.stack(latents, dim=1)         # [B, T+1, C,H,W]
            log_probs = torch.stack(log_probs, dim=1)     # [B, T]
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.train_batch_size, 1).to(accelerator.device)

            x0 = latents[:, -1]                           # clean latent from trajectory

            score_result = compute_intrinsic_score_map(
                transformer=pipeline.transformer,
                prompts=prompts,
                x0=x0,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_encoders=text_encoders,
                tokenizers=tokenizers,
                neg_prompt_embed=neg_prompt_embed,
                neg_pooled_prompt_embed=neg_pooled_prompt_embed,
                config=config,
                device=accelerator.device,
                use_steps=(
                    num_train_timesteps
                    if config.cf.score_type == "raw"
                    else get_grpo_probe_step_indices(num_train_timesteps)
                ),
                autocast_ctx=autocast,
            )

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],       # latent before t
                    "next_latents": latents[:, 1:],   # latent after t
                    "log_probs": log_probs,
                    "rewards": {
                        "avg": score_result["selected_score"],
                        **{
                            key: value
                            for key, value in score_result["scores"].items()
                            if key in {"raw", "pmi", "cope", "cope_lse"}
                        },
                    },
                    "reward_per_step": score_result["selected_per_step"],
                }
            )
            prompts_last = prompts
            images_last = images

        # Collate to big tensors
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0) for sub_key in samples[0][k]}
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process and images_last is not None and experiment_logger is not None:
            num_samples = min(15, len(images_last))
            sample_indices = random.sample(range(len(images_last)), num_samples)
            sampled_prompts = [prompts_last[ii] for ii in sample_indices]
            sampled_rewards = [samples["rewards"]["avg"][ii].item() for ii in sample_indices]
            experiment_logger.log_images(
                "train_images",
                [images_last[ii].detach().cpu() for ii in sample_indices],
                step=global_step,
                captions=[f"{prompt:.100} | reward: {avg_reward:.2f}" for prompt, avg_reward in zip(sampled_prompts, sampled_rewards)],
            )

        # Keep original keys for downstream stats
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"].clone()
        samples["rewards"]["reward_per_step"] = samples["reward_per_step"].clone()
        # Expand the scalar reward along time for GRPO
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)

        # Gather across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        if accelerator.is_local_main_process:
            print(
                f"Reward stats: max ({gathered_rewards['reward_per_step'].max()}) "
                f"min ({gathered_rewards['reward_per_step'].min()}) "
                f"mean({gathered_rewards['reward_per_step'].mean()})"
            )

        if accelerator.is_main_process and experiment_logger is not None:
            experiment_logger.log_scalars(
                {"epoch": epoch, **{f"reward/{key}": value.mean() for key, value in gathered_rewards.items()}},
                step=global_step,
            )

        # Per-prompt stat tracking on the selected intrinsic reward
        if config.per_prompt_stat_tracking:
            prompt_ids_all = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts_all = pipeline.tokenizer.batch_decode(prompt_ids_all, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts_all, gathered_rewards['avg'])
            if accelerator.is_main_process:
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all, gathered_rewards)
                if experiment_logger is not None:
                    experiment_logger.log_scalars(
                        {
                            "group_size": group_size,
                            "trained_prompt_num": trained_prompt_num,
                            "zero_std_ratio": zero_std_ratio,
                            "reward_std_mean": reward_std_mean,
                        },
                        step=global_step,
                    )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # Ungather advantages back to local shard
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]
        del samples["reward_per_step"]

        # Mask zero-adv examples across T, keep batch divisibility for num_batches
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process and experiment_logger is not None:
            experiment_logger.log_scalars({"actual_batch_size": mask.sum().item() // num_batches}, step=global_step)
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps

        # ========== TRAINING ==========
        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            samples_batched = {
                k: v.reshape(-1, total_batch_size // config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                train_timesteps = [step_index for step_index in range(int(num_train_timesteps * 0.5), num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
                                transformer, pipeline, sample, j, embeds, pooled_embeds, config
                            )
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with adapter_off_ctx(transformer):
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(
                                            transformer, pipeline, sample, j, embeds, pooled_embeds, config
                                        )

                        # GRPO with intrinsic-reward-derived advantages
                        advantages_j = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages_j * ratio
                        clipped_loss = -advantages_j * torch.clamp(
                            ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                        info["clipfrac_gt_one"].append(torch.mean((ratio - 1.0 > config.train.clip_range).float()))
                        info["clipfrac_lt_one"].append(torch.mean((1.0 - ratio > config.train.clip_range).float()))
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)
                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process and experiment_logger is not None:
                            experiment_logger.log_scalars(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)

        epoch += 1

    if experiment_logger is not None:
        experiment_logger.close()
    executor.shutdown(wait=False)

if __name__ == "__main__":
    app.run(main)
