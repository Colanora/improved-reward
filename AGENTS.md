# Repository Guidelines

## Project Structure & Module Organization
`solace/` contains the Python package: reward functions, prompt helpers, utilities, and model-specific patches in `solace/diffusers_patch/`. `config/` holds the shared base config and named experiment presets selected with `--config config/solace.py:<name>`. `scripts/` contains training entrypoints plus launch helpers in `scripts/single_node/` and `scripts/accelerate_configs/`. `dataset/` stores prompt corpora and evaluation metadata such as `train.txt`, `test.txt`, and `*_metadata.jsonl`.

## Build, Test, and Development Commands
`pip install -e .` installs the package for local development. `pip install -e .[dev]` adds `black` and `pytest`.

`python -m black solace config scripts` formats Python sources with the repo’s declared formatter.

`pytest` is the default test command, but there is no committed `tests/` suite yet; add focused tests when introducing pure-Python logic.

`bash scripts/single_node/grpo_self.sh` launches the default 8-GPU SD3.5 training job. For custom runs, use the same pattern as the scripts, for example:
```bash
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=1 scripts/train_sd3_self.py \
  --config config/solace.py:general_ocr_sd3_1gpu
```

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and concise inline comments only where the control flow is non-obvious. Keep new training entrypoints named consistently with the current pattern, such as `scripts/train_<model>_self.py`. Add new config factories in `config/solace.py` with descriptive names like `general_ocr_sd3_8gpu`.

## Testing Guidelines
Prefer `pytest` for new unit coverage, especially around helpers in `solace/` that do not require GPU execution. Name tests `test_<module>.py`. For training-path changes, include a smoke-tested command and config in the PR description, ideally using a 1-GPU or reduced-batch config before scaling to multi-GPU runs.

## Commit & Pull Request Guidelines
Git history is currently minimal and starts with `init`, so no strong convention is established. Use short, imperative commit subjects and include a scope when useful, for example `config: tune sd3 1gpu defaults`. PRs should describe the affected model or reward path, list changed configs or datasets, note expected hardware, and attach sample metrics or generated outputs when behavior changes.

## Configuration & Artifacts
Do not commit training artifacts. `.gitignore` already excludes `logs/`, `wandb/`, `outputs/`, and checkpoint-like output folders. Keep secrets and mirror endpoints in environment variables rather than hardcoding them in scripts.
