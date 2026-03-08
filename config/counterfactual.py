import importlib.util
import os

import ml_collections


def _load_base_module():
    base_path = os.path.join(os.path.dirname(__file__), "base.py")
    spec = importlib.util.spec_from_file_location("solace_counterfactual_base", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config base module from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()


def get_config(config_string: str = ""):
    if not config_string:
        return sd3_cf_rerank_2gpu()

    config_fn = globals().get(config_string)
    if config_fn is None or not callable(config_fn):
        raise ValueError(f"Unknown counterfactual config: {config_string}")
    return config_fn()


def sd3_cf_rerank_2gpu():
    config = base.get_config()
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    config.resolution = 512
    config.mixed_precision = "bf16"
    config.seed = 42

    config.sample.num_steps = 10
    config.sample.eval_num_steps = 10
    config.sample.guidance_scale = 7.0

    config.cf = ml_collections.ConfigDict()
    config.cf.num_candidates = 8
    config.cf.num_probe_steps = 5
    config.cf.k = 8
    config.cf.delta = 1e-6
    config.cf.score_type = "cope"
    config.cf.negative_mode = "auto"
    config.cf.num_negatives = 1
    config.cf.use_cfg_probe = False
    config.cf.normalize_per_step = True
    config.cf.shared_probes = True
    config.cf.time_weighting = "mid"
    config.cf.metrics = ["ocr"]
    config.cf.dataset_split = "test"
    config.cf.save_all_candidates = True
    config.cf.save_latents = False
    config.cf.max_prompts = 0
    config.cf.seed_stride = 1000

    config.output_dir = "logs/cf_rerank_sd3"
    return config


def sd3_cf_structured_rerank_2gpu():
    config = sd3_cf_rerank_2gpu()
    config.dataset = os.path.join(os.getcwd(), "dataset/cf_structured")
    config.cf.metrics = []
    config.output_dir = "logs/cf_rerank_sd3_structured"
    return config


def sd3_cf_rerank_2gpu_fp16():
    config = sd3_cf_rerank_2gpu()
    config.mixed_precision = "fp16"
    config.output_dir = "logs/cf_rerank_sd3_fp16"
    return config


def sd3_cf_structured_rerank_2gpu_fp16():
    config = sd3_cf_structured_rerank_2gpu()
    config.mixed_precision = "fp16"
    config.output_dir = "logs/cf_rerank_sd3_structured_fp16"
    return config


def sd3_cf_dpo_2gpu():
    config = sd3_cf_rerank_2gpu()
    config.train = ml_collections.ConfigDict()
    config.train.learning_rate = 1e-4
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 4
    config.train.max_steps = 3000
    config.train.eval_every = 200
    config.train.save_every = 500
    config.train.max_grad_norm = 1.0
    config.train.use_lora = True
    config.train.use_8bit_adam = False
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 1e-4
    config.train.adam_epsilon = 1e-8
    config.train.lora_rank = 32
    config.train.lora_alpha = 64
    config.train.lora_dropout = 0.0
    config.train.lora_path = None
    config.train.target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]

    config.dpo = ml_collections.ConfigDict()
    config.dpo.beta = 0.1
    config.dpo.margin_threshold = 0.25
    config.dpo.safeguarded = False
    config.output_dir = "logs/cf_dpo_sd3"
    return config
