import datetime
import inspect
import os

from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from ml_collections import config_flags
from peft import LoraConfig, PeftModel, get_peft_model
import torch
from torch.utils.data import DataLoader

from solace.counterfactual_sd3_utils import load_sd3_pipeline, score_prompt_candidates
from solace.dpo_utils import (
    CounterfactualPairDataset,
    load_terminal_latent,
    preference_accuracy,
    preference_loss,
)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/counterfactual.py:sd3_cf_dpo_2gpu", "Counterfactual SD3 DPO config.")
flags.DEFINE_string("pairs_jsonl", None, "Path to the mined pseudo-pair JSONL.")
flags.DEFINE_string("val_pairs_jsonl", None, "Optional validation pseudo-pair JSONL.")
flags.DEFINE_string("output_dir", None, "Optional output directory override.")
logger = get_logger(__name__)


def _build_optimizer(config, parameters):
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("Install bitsandbytes for 8-bit Adam: `pip install bitsandbytes`") from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    return optimizer_cls(
        parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )


def _attach_lora(transformer, config):
    lora_config = LoraConfig(
        r=config.train.lora_rank,
        lora_alpha=config.train.lora_alpha,
        lora_dropout=config.train.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=list(config.train.target_modules),
    )
    if config.train.lora_path:
        load_signature = inspect.signature(PeftModel.from_pretrained)
        if "is_trainable" in load_signature.parameters:
            return PeftModel.from_pretrained(
                transformer,
                config.train.lora_path,
                is_trainable=True,
            )

        model = PeftModel.from_pretrained(transformer, config.train.lora_path)
        if hasattr(model, "enable_adapter_layers"):
            model.enable_adapter_layers()
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True
        return model
    return get_peft_model(transformer, lora_config)


def _score_pair(example, pipeline, text_encoders, tokenizers, config, device):
    negative_prompts = example.get("negative_prompts") or [example.get("negative_prompt", "")]
    if not negative_prompts:
        negative_prompts = [""]

    winner_latent = load_terminal_latent(example["winner_latent_path"], device=device)
    loser_latent = load_terminal_latent(example["loser_latent_path"], device=device)
    x0 = torch.stack([winner_latent, loser_latent], dim=0).to(device)
    timesteps = pipeline.scheduler.timesteps.repeat(2, 1).to(device)

    score_result = score_prompt_candidates(
        transformer=pipeline.transformer,
        prompt=example["prompt"],
        x0=x0,
        timesteps=timesteps,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        config=config,
        device=device,
        negative_prompts_override=negative_prompts,
        negative_mode_override=example.get("negative_mode"),
    )

    score_type = example.get("score_type", config.cf.score_type)
    scores = score_result["scores"][score_type]
    return scores[0], scores[1]


def _evaluate(dataloader, pipeline, text_encoders, tokenizers, config, accelerator):
    losses = []
    accuracies = []
    for batch in dataloader:
        batch_loss = []
        batch_accuracy = []
        with torch.no_grad():
            for example in batch:
                chosen_score, rejected_score = _score_pair(
                    example=example,
                    pipeline=pipeline,
                    text_encoders=text_encoders,
                    tokenizers=tokenizers,
                    config=config,
                    device=accelerator.device,
                )
                batch_loss.append(
                    preference_loss(
                        chosen_score=chosen_score,
                        rejected_score=rejected_score,
                        beta=config.dpo.beta,
                        margin_threshold=config.dpo.margin_threshold,
                        safeguarded=config.dpo.safeguarded,
                    )
                )
                batch_accuracy.append(
                    preference_accuracy(
                        chosen_score=chosen_score,
                        rejected_score=rejected_score,
                        margin_threshold=config.dpo.margin_threshold,
                    )
                )

        if batch_loss:
            losses.append(torch.stack(batch_loss).mean())
            accuracies.append(torch.stack(batch_accuracy).mean())

    if not losses:
        return None

    mean_loss = torch.stack(losses).mean()
    mean_accuracy = torch.stack(accuracies).mean()
    gathered_loss = accelerator.gather_for_metrics(mean_loss.unsqueeze(0)).mean().item()
    gathered_accuracy = accelerator.gather_for_metrics(mean_accuracy.unsqueeze(0)).mean().item()
    return {"val_loss": gathered_loss, "val_accuracy": gathered_accuracy}


def main(_):
    if not FLAGS.pairs_jsonl:
        raise ValueError("--pairs_jsonl is required.")

    config = FLAGS.config
    if FLAGS.output_dir is not None:
        config.output_dir = FLAGS.output_dir

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    device = accelerator.device
    set_seed(config.seed, device_specific=True)

    run_id_holder = [datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if accelerator.is_main_process else None]
    if accelerator.num_processes > 1:
        torch.distributed.broadcast_object_list(run_id_holder, src=0)
    run_id = run_id_holder[0]
    run_dir = os.path.join(config.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    pipeline, text_encoders, tokenizers = load_sd3_pipeline(
        config=config,
        device=device,
        is_local_main_process=accelerator.is_local_main_process,
    )
    pipeline.scheduler.set_timesteps(config.sample.eval_num_steps, device=device)
    pipeline.transformer.requires_grad_(False)
    pipeline.transformer = _attach_lora(pipeline.transformer, config)
    pipeline.transformer.set_adapter("default")
    trainable_parameters = [parameter for parameter in pipeline.transformer.parameters() if parameter.requires_grad]

    optimizer = _build_optimizer(config, trainable_parameters)

    train_dataset = CounterfactualPairDataset(FLAGS.pairs_jsonl)
    if len(train_dataset) == 0:
        raise ValueError("The pseudo-pair dataset is empty.")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=CounterfactualPairDataset.collate_fn,
    )

    val_dataloader = None
    if FLAGS.val_pairs_jsonl:
        val_dataset = CounterfactualPairDataset(FLAGS.val_pairs_jsonl)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=CounterfactualPairDataset.collate_fn,
        )

    pipeline.transformer, optimizer, train_dataloader = accelerator.prepare(
        pipeline.transformer,
        optimizer,
        train_dataloader,
    )
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    global_step = 0
    stop_training = False
    while not stop_training:
        for batch in train_dataloader:
            pipeline.transformer.train()
            with accelerator.accumulate(pipeline.transformer):
                losses = []
                accuracies = []
                for example in batch:
                    chosen_score, rejected_score = _score_pair(
                        example=example,
                        pipeline=pipeline,
                        text_encoders=text_encoders,
                        tokenizers=tokenizers,
                        config=config,
                        device=device,
                    )
                    losses.append(
                        preference_loss(
                            chosen_score=chosen_score,
                            rejected_score=rejected_score,
                            beta=config.dpo.beta,
                            margin_threshold=config.dpo.margin_threshold,
                            safeguarded=config.dpo.safeguarded,
                        )
                    )
                    accuracies.append(
                        preference_accuracy(
                            chosen_score=chosen_score,
                            rejected_score=rejected_score,
                            margin_threshold=config.dpo.margin_threshold,
                        )
                    )

                loss = torch.stack(losses).mean()
                accuracy = torch.stack(accuracies).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipeline.transformer.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                reduced_loss = accelerator.gather_for_metrics(loss.detach().unsqueeze(0)).mean().item()
                reduced_accuracy = accelerator.gather_for_metrics(accuracy.detach().unsqueeze(0)).mean().item()
                logger.info("step=%s loss=%.4f accuracy=%.4f", global_step, reduced_loss, reduced_accuracy)

                if global_step % config.train.save_every == 0 and accelerator.is_main_process:
                    save_dir = os.path.join(run_dir, f"checkpoint-{global_step}")
                    accelerator.unwrap_model(pipeline.transformer).save_pretrained(save_dir)

                if val_dataloader is not None and global_step % config.train.eval_every == 0:
                    pipeline.transformer.eval()
                    eval_metrics = _evaluate(
                        dataloader=val_dataloader,
                        pipeline=pipeline,
                        text_encoders=text_encoders,
                        tokenizers=tokenizers,
                        config=config,
                        accelerator=accelerator,
                    )
                    if accelerator.is_main_process and eval_metrics is not None:
                        logger.info("validation step=%s %s", global_step, eval_metrics)

                if global_step >= config.train.max_steps:
                    stop_training = True
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(pipeline.transformer).save_pretrained(os.path.join(run_dir, "final_lora"))


if __name__ == "__main__":
    app.run(main)
