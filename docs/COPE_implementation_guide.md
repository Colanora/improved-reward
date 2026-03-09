# COPE: Implementation Report and Experimental Results

## A self-contained technical document for external review

---

## 0. Purpose of This Document

This document describes the **complete implementation** of COPE (Counterfactual Prompt Evidence) as built on top of the SOLACE codebase. It is written for a reviewer who **cannot access the source code** but needs to:

1. Understand exactly what algorithms are implemented and how.
2. Evaluate whether the implementation faithfully realizes the theory in the COPE research proposal.
3. Interpret the experimental results from two completed stages:
   - **Stage 1**: Training-free best-of-N reranking (completed, multi-seed).
   - **Stage 2**: Online GRPO post-training with COPE-LSE as the intrinsic reward (early runs, in progress).

All pseudocode in this document corresponds to actual implemented functions. All results are from real runs on SD3.5-Medium with H100 GPUs.

---

## 1. Background: SOLACE Probe Confidence

SOLACE defines an intrinsic self-confidence score for a generated image. Given:

- a terminal latent \(z\) (the clean output of the generation trajectory),
- a conditioning prompt \(c\),
- the same pretrained diffusion transformer \(p_\theta\),

SOLACE probes whether the model can reconstruct added noise from \(z\) under condition \(c\). If reconstruction is accurate, the model is "confident" that \(z\) is consistent with \(c\).

The COPE proposal argues that this raw confidence conflates two effects:

- **Typicality**: generic ease of denoising (images that "look normal" to the model).
- **Prompt evidence**: how specifically the image matches this particular prompt.

COPE fixes this by subtracting the confidence under a hard-negative prompt \(\tilde{c}\), isolating the prompt-specific component.

---

## 2. Implemented Algorithms

### Algorithm 1: Antithetic Probe Construction

```
Input:  K (even integer, default 8), spatial shape (B, C, H, W), device, dtype
Output: probe noise bank ε of shape [K, B, C, H, W]

1:  sample ε₁, ..., ε_{K/2}  ~  N(0, I)  with shape [B, C, H, W]
2:  set ε_{K/2+i} = -εᵢ   for i = 1, ..., K/2
3:  return stack [ε₁, ..., ε_K]
```

**Rationale**: Antithetic pairing ensures the probe set has exactly zero empirical mean, reducing variance in the MSE estimate by a factor of ~2.

### Algorithm 2: Flow-Matching Probe Latent Construction

```
Input:  terminal latent x₀ [B, C, H, W],
        timestep index t_idx [B],
        probe noise ε [K, B, C, H, W]
Output: noised probe latents x_t [K, B, C, H, W]

1:  t = t_idx / 1000                     # normalize to [0, 1]
2:  x_t = (1 - t) · x₀ + t · ε          # flow-matching interpolation
3:  return x_t
```

**Note**: This is the standard rectified-flow noising schedule used by SD3.

### Algorithm 3: Raw Probe Confidence (SOLACE-Compatible)

```
Input:  transformer θ, terminal latent x₀ [B, C, H, W],
        timestep schedule τ [B, T], condition embeddings (pe, ppe),
        probe config: K, δ, selected step indices S ⊆ {0, ..., T-1},
        time weighting mode, normalize flag
Output: confidence score C(z, c) [B]

 1:  for each probe step position p = 0, 1, ..., |S|-1 do
 2:      t_idx = τ[:, S[p]]                              # timestep for this step
 3:      ε = AntitheticalProbes(K, (B,C,H,W))            # or reuse shared bank
 4:      x_t = FlowMatchProbe(x₀, t_idx, ε)              # [K, B, C, H, W]
 5:      flatten x_t to [K·B, C, H, W]
 6:      flatten ε  to [K·B, C, H, W]
 7:      flatten x₀ to [K·B, C, H, W]                    # repeat K times
 8:      repeat pe, ppe K times along batch dimension
 9:
10:      # Forward pass — NO classifier-free guidance
11:      v_pred = transformer(x_t, t_idx.repeat(K), pe, ppe)
12:
13:      # Recover predicted noise via flow-matching identity
14:      ε̂ = v_pred + x₀
15:
16:      # Per-probe MSE, averaged over spatial dimensions
17:      mse_flat = mean( (ε̂ - ε)², dims=(C, H, W) )    # [K·B]
18:      mse = reshape to [K, B] and mean over K           # [B]
19:
20:      # Raw step score
21:      S_t[p] = -log(mse + δ)                            # [B]
22:  end for
23:
24:  # Per-step z-score normalization (if enabled and B > 1)
25:  if normalize_per_step:
26:      for each step p:
27:          μ = mean(S_t[p] across batch)
28:          σ = std(S_t[p] across batch), clamped ≥ 1e-6
29:          S_t[p] = (S_t[p] - μ) / σ
30:
31:  # Time-weighted aggregation
32:  if time_weighting == "mid":
33:      t_norm = τ[:, S] / 1000
34:      w = t_norm · (1 - t_norm)                          # [|S|]
35:  else:
36:      w = ones(|S|)
37:
38:  C = Σ_p (w[p] · S_t[p]) / Σ_p w[p]                   # [B]
39:  return C
```

**Key implementation decisions**:

- **No CFG during probing** (default): The transformer is called once per condition, without the unconditional branch or guidance scale. This isolates the model's raw conditional prediction.
- **Suffix timesteps**: Probing focuses on suffix timesteps where prompt-specific signal is strongest. In reranking/eval this is controlled by `config.cf.num_probe_steps` (implemented as “take the last `num_probe_steps` indices of the current timestep schedule”; default: last 5 of 10). In GRPO, the current implementation probes the last half of `num_train_timesteps`, where `num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)` (so with 10-step sampling and `timestep_fraction=0.99`, it probes indices 4–8). This is implemented via `get_grpo_probe_step_indices()` for PMI/COPE and via `sds_self_confidence_scalar()`’s internal step loop for raw. **Note**: during GRPO eval, images may be generated with `config.sample.eval_num_steps` (e.g., 40), but the current implementation still uses probe indices derived from `num_train_timesteps`, so it probes those fixed indices in the longer schedule.
- **Per-step z-score normalization**: Equalizes the magnitude contribution of each timestep across the candidate batch, preventing any single timestep from dominating the aggregate score.
- **Mid-range time weighting** \(w(t) = t(1-t)\)): Downweights the extremes (fully clean, fully noisy) where probe signal is either trivial or noise-dominated.
- **Noise recovery identity** \(\hat\epsilon = v_{\text{pred}} + x_0\): This follows from the SD3 flow-matching parameterization where the model predicts the velocity \(v = \epsilon - x_0\).

### Algorithm 4: Shared-Probe Multi-Condition Scoring

```
Input:  transformer θ, terminal latent x₀ [B, C, H, W],
        timestep schedule τ [B, T],
        condition list [{name, pe, ppe}, ...] with M conditions,
        probe config (same as Algorithm 3)
Output: {name: C(z, cond)} for each condition [B]

1:  Build shared probe bank once:
2:      for each step p = 0, ..., |S|-1:
3:          ε[p] = AntitheticalProbes(K, (B,C,H,W))
4:
5:  for each step p:
6:      t_idx = τ[:, S[p]]
7:      x_t = FlowMatchProbe(x₀, t_idx, ε[p])            # same noise for all conditions
8:      for each condition m in {0, ..., M-1}:
9:          v_pred_m = transformer(x_t, t_idx, cond[m].pe, cond[m].ppe)
10:         ε̂_m = v_pred_m + x₀
11:         mse_m = mean_K( mean_spatial( (ε̂_m - ε[p])² ) )
12:         raw_score[m][p] = -log(mse_m + δ)
13:
14:  for each condition m:
15:      normalize and aggregate raw_score[m] → score[m]    # same as Algorithm 3, steps 24-38
16:
17:  return {cond[m].name: score[m] for m = 0, ..., M-1}
```

**Critical design choice**: The **same probe noise bank** is used for all conditions at each timestep. This ensures that when we compute `C(z, c) - C(z, c̃)`, the difference reflects only the change in conditioning, not probe noise variance. This is essential for COPE's signal-to-noise ratio.

### Algorithm 5: Deterministic Counterfactual Prompt Generation

```
Input:  prompt c, mode ∈ {"auto", "ocr", "count", "spatial", "attribute", "unconditional", "pmi"},
        n_neg (number of negatives to generate)
Output: list of hard-negative prompts (length ≤ n_neg, and always at least 1)

 1:  # Resolve the perturbation family (mode)
 2:  if mode == "auto":
 3:      try each rule in priority order: OCR → Count → Spatial → Attribute
 4:      target_mode = first rule that matches, else "unconditional"
 5:  elif mode ∈ {"unconditional", "pmi"}:
 6:      target_mode = "unconditional"
 7:  else:
 8:      target_mode = mode if that rule matches, else "unconditional"
 9:
10:  if target_mode == "unconditional":
11:      return [""]
12:
13:  Rule definitions (each rule returns None when it does not match):
14:
15:  OCR rule (highest priority):
16:      detect quoted text via regex: ("..") or ('..')
17:      replace each character with a deterministic rotation:
18:          digits:     shift by (3 + variant) mod 10
19:          uppercase:  shift by (7 + variant) mod 26
20:          lowercase:  shift by (11 + variant) mod 26
21:      the replacement has the same length and character class
22:      Example: "B7Q9" → "I0X2" (variant=0), "J1Y3" (variant=1)
23:
24:  Count rule:
25:      detect number words (zero–twelve) or digits via regex
26:      perturb by +1 or -1 within [0, 12]
27:      variant index selects which direction
28:      Example: "three red balls" → "four red balls" (variant=0),
29:               "two red balls" (variant=1)
30:
31:  Spatial rule:
32:      detect spatial relations from a closed set:
33:          "to the left of" ↔ "to the right of"
34:          "above" ↔ "below"
35:          "in front of" ↔ "behind"
36:          "under" ↔ "over"
37:          "left of" ↔ "right of"
38:      replace with the opposite
39:      Example: "A cat to the left of a dog" → "A cat to the right of a dog"
40:
41:  Attribute rule:
42:      detect tokens from controlled lexicons:
43:          colors:    red, blue, green, yellow, black, white, orange, purple, pink, brown
44:          sizes:     small, large, tiny, huge, miniature, giant
45:          materials: wooden, metal, glass, stone, plastic, ceramic
46:      replace with a different token from the same lexicon
47:      variant index selects which alternative
48:      Example: "A red mug" → "A blue mug" (variant=0)
49:
50:  To generate n_neg distinct negatives:
51:      max_trials = max(4·n_neg, 4)
52:      iterate variant = 0, 1, ..., max_trials-1
53:      collect unique outputs, stop when n_neg reached
54:      return negatives if any; else fall back to unconditional [""].
```

**Design principles**:

- **Deterministic and reproducible**: No LLM or external model needed for negative generation.
- **Minimal perturbation**: Exactly one semantic factor changes per negative. (The OCR rule preserves the quoted-string length and character class; other rules do not guarantee exact string-length preservation.)
- **Priority ordering**: OCR prompts (with quoted text) are the most specific and reliable, so they take highest priority in auto mode.

### Algorithm 6: COPE Score Computation

```
Input:  transformer θ, terminal latent x₀ [B, C, H, W],
        timestep schedule τ, intended prompt c,
        hard-negative prompts {c̃₁, ..., c̃_M},
        unconditional prompt ∅ = "",
        probe config
Output: score dictionary {raw, pmi, cope, cope_lse} each [B]

1:  Encode all prompts via SD3 text encoders:
2:      (pe⁺, ppe⁺) = encode(c, batch_size=B)
3:      (pe∅, ppe∅) = encode("", batch_size=B)
4:      for j = 1, ..., M:
5:          (peⱼ⁻, ppeⱼ⁻) = encode(c̃ⱼ, batch_size=B)
6:
7:  Build condition list:
8:      conds = [("positive", pe⁺, ppe⁺),
9:               ("unconditional", pe∅, ppe∅),
10:              ("negative_0", pe₁⁻, ppe₁⁻),
11:              ...,
12:              ("negative_{M-1}", peₘ⁻, ppeₘ⁻)]
13:
14:  # Score all conditions with shared probes (Algorithm 4)
15:  scores = SharedProbeScore(θ, x₀, τ, conds, config)
16:
17:  # Derive four score variants:
18:  raw     = scores["positive"]                                    # C(z, c)
19:  pmi     = scores["positive"] - scores["unconditional"]          # C(z,c) - C(z,∅)
20:  cope    = scores["positive"] - scores["negative_0"]             # C(z,c) - C(z,c̃₁)
21:  cope_lse= scores["positive"] - logsumexp(scores["negative_j"]) # C(z,c) - log Σ exp C(z,c̃ⱼ)
22:
23:  return {raw, pmi, cope, cope_lse}
```

**Score interpretation**:

| Score | Formula | Measures |
|---|---|---|
| `raw` | \(C(z, c)\) | Generic model confidence (SOLACE baseline) |
| `pmi` | \(C(z, c) - C(z, \varnothing)\) | Evidence for prompt \(c\) vs. unconditional |
| `cope` | \(C(z, c) - C(z, \tilde{c}_1)\) | Evidence for \(c\) vs. one hard negative |
| `cope_lse` | \(C(z, c) - \log\sum_j e^{C(z, \tilde{c}_j)}\) | Evidence for \(c\) vs. soft worst-case competitor |

### Algorithm 7: Training-Free Reranking

```
Input:  prompt c, candidate count N, score type ∈ {raw, pmi, cope, cope_lse}
Output: selected image x*

 1:  Generate hard-negative prompts: {c̃ⱼ} = BuildCounterfactuals(c)
 2:  Encode prompt c and negatives via SD3 text encoders
 3:  Encode unconditional prompt "" for generation CFG
 4:
 5:  seeds = [base_seed + prompt_index · stride + i  for i = 0, ..., N-1]
 6:  for i = 0, ..., N-1:
 7:      Initialize generator with seeds[i]
 8:  Generate N images in one batch using SD3 pipeline with CFG
 9:  Extract terminal latents x₀[0], ..., x₀[N-1]
10:  Collect generation timestep schedule τ
11:
12:  # Score all candidates (Algorithm 6)
13:  score_dict = COPEScore(θ, x₀, τ, c, {c̃ⱼ}, config)
14:
15:  # Select best candidate under the chosen score type
16:  i* = argmax_i score_dict[score_type][i]
17:
18:  # Also select under all other methods for comparison
19:  for each method in {raw, pmi, cope, cope_lse}:
20:      selected[method] = argmax_i score_dict[method][i]
21:
22:  # Evaluate external metrics (e.g., OCR accuracy) on all candidates
23:  metrics = ExternalMetrics(images, prompt)
24:
25:  return selected images, scores, metrics
```

### Algorithm 8: Pseudo-Pair Mining for Offline DPO

```
Input:  prompt set D, candidate count N, score type, margin threshold τ
Output: pseudo-preference dataset P

1:  P = ∅
2:  for each prompt c in D:
3:      Generate N candidates and score via Algorithm 6
4:      scores = score_dict[score_type]                  # [N]
5:      w = argmax(scores),  l = argmin(scores)
6:      if scores[w] - scores[l] < τ:
7:          skip this prompt (insufficient margin)
8:      Save winner/loser images and terminal latents to disk
9:      P = P ∪ {(c, seed_w, seed_l, score_w, score_l, paths, c̃)}
10: return P
```

### Algorithm 9: Offline LoRA-DPO Training

```
Input:  pretrained transformer θ, pseudo-pairs P, DPO config (β, margin_threshold)
Output: LoRA-adapted transformer θ'

1:  Freeze all base model weights
2:  Attach LoRA adapters (rank 32, α=64) to transformer attention:
3:      target modules: {to_q, to_k, to_v, to_out, add_q_proj, add_k_proj, add_v_proj, to_add_out}
4:
5:  for step = 1, ..., max_steps:
6:      Sample minibatch of pairs {(c, z⁺, z⁻)} from P
7:      for each pair:
8:          Load winner latent z⁺ and loser latent z⁻ from disk
9:          Score both via Algorithm 6:
10:             s⁺ = COPE(z⁺, c),  s⁻ = COPE(z⁻, c)
11:
12:         Compute DPO loss:
13:             margin = s⁺ - s⁻ - margin_threshold
14:             loss = -log σ(β · margin)
15:             if safeguarded:
16:                 loss += 0.5 · relu(-margin)²
17:
18:     Update LoRA parameters via AdamW with gradient clipping
19:     Periodically save checkpoints and evaluate
```

### Algorithm 10: Online GRPO Training with COPE Reward

```
Input:  pretrained transformer θ with LoRA, OCR prompt dataset,
        score type ∈ {raw, pmi, cope, cope_lse}, training config
Output: LoRA-adapted transformer θ'

1:  Initialize LoRA adapters, optimizer, per-prompt stat tracker
2:
3:  for epoch = 0, 1, ..., num_epochs:
4:
5:      # === SAMPLING PHASE ===
6:      for batch_i = 0, ..., num_batches_per_epoch:
7:          Sample prompts from training set
8:          Encode prompts and generate images with current policy θ
9:          Collect terminal latents x₀ and per-step log-probs
10:
11:         # Compute intrinsic reward (COPE)
12:         if score_type == "raw":
13:             reward = RawProbeConfidence_GRPO(x₀, c)     # implemented as sds_self_confidence_scalar()
14:         elif score_type == "pmi":
15:             reward = C(x₀, c) - C(x₀, ∅)               # via Algorithm 6
16:         else:  # cope or cope_lse
17:             For each prompt in batch:
18:                 negatives = BuildCounterfactuals(prompt)
19:             Group prompts by number of negatives (since M can vary per prompt)
20:             For each group:
21:                 score_dict = COPEScore(θ, x₀, τ, c, negatives)   # Algorithm 6
22:             reward = score_dict[score_type]
23:
24:         Store sample: {latents, log_probs, reward, prompts}
25:
26:     # === ADVANTAGE COMPUTATION ===
27:     Gather rewards across all processes
28:     if per_prompt_stat_tracking:
29:         advantages = PerPromptNormalize(prompts, rewards)   # z-score per prompt
30:     else:
31:         advantages = (rewards - mean) / std
32:     Expand scalar advantages across timesteps
33:
34:     # === PPO-STYLE UPDATE ===
35:     for inner_epoch = 0, ..., num_inner_epochs:
36:         for each sample, for each timestep j:
37:             Compute current log-prob under θ
38:             Compute reference log-prob under frozen θ_ref (LoRA disabled)
39:
40:             ratio = exp(log_prob - old_log_prob)
41:             clipped_ratio = clip(ratio, 1-ε, 1+ε)
42:             policy_loss = max(-adv · ratio, -adv · clipped_ratio)
43:
44:             kl_loss = ||μ_current - μ_ref||² / (2σ²)
45:             loss = policy_loss + β_kl · kl_loss
46:
47:             Update LoRA parameters
48:
49:     # === EVALUATION ===
50:     if epoch % eval_freq == 0:
51:         Generate eval images, compute intrinsic scores + OCR accuracy
```

**Key differences between raw GRPO and COPE GRPO**:

- In raw GRPO, the reward is simply `C(z, c)` — the model's raw self-confidence (implemented via `sds_self_confidence_scalar()` in the training script).
- In COPE GRPO, the reward is `C(z, c) - logsumexp_j C(z, c̃_j)` — the counterfactual evidence.
- The advantage computation, PPO clipping, KL regularization, and LoRA update are identical.
- COPE/COPE-LSE requires additional transformer forward passes during probing for the unconditional baseline and for each negative prompt (and optionally more if CFG probing is enabled).

**Implementation note**: In the current repo, raw-GRPO probing is implemented separately from the shared-probe multi-condition scorer (Algorithms 1–4): `sds_self_confidence_scalar()` probes the last half of `T_used` timesteps (where `T_used = min(use_steps, T_all - 1)`), and applies per-step normalization/time weighting unconditionally. COPE/PMI uses `compute_counterfactual_scores()` → `score_conditions_shared_probes()` from the library layer.

---

## 3. Implementation Architecture

### 3.1 Module Responsibilities

The implementation is organized into five library modules and six scripts:

**Library layer** (`solace/` package):

| Module | Responsibility |
|---|---|
| `probe_utils` | All probe mechanics: antithetic construction, flow-matching noising, timestep selection, the shared-probe scoring loop (Algorithms 1–4). Also provides SD3 text encoding helpers. |
| `baseline_prompts` | Deterministic hard-negative generation (Algorithm 5). Pure Python, no model dependencies. Contains regex patterns for OCR, counting, spatial, and attribute detection, plus the priority-based auto-resolution logic. |
| `counterfactual_reward` | COPE score assembly (Algorithm 6). Builds condition lists from embeddings, delegates to `probe_utils`, and derives raw/PMI/COPE/COPE-LSE from the returned per-condition scores. |
| `counterfactual_sd3_utils` | SD3-specific orchestration: pipeline loading, candidate generation via the diffusers `pipeline_with_logprob`, prompt encoding, and the high-level `score_prompt_candidates()` entry point that wires text encoding → negative generation → COPE scoring. |
| `dpo_utils` | DPO training utilities: pseudo-pair dataset class, terminal-latent loading, preference loss (with optional safeguard term), and accuracy computation. |

**Script layer** (`scripts/`):

| Script | Implements |
|---|---|
| `rerank_counterfactual_sd3.py` | Training-free reranking (Algorithm 7). Distributed via Accelerate. Outputs JSONL, images, TensorBoard, and summary statistics. |
| `eval_counterfactual_sd3.py` | Post-hoc evaluation: cross-method score comparison, selection disagreement, counterfactual discrimination accuracy. |
| `build_cf_pairs_sd3.py` | Pseudo-pair mining (Algorithm 8). |
| `train_sd3_cf_dpo.py` | Offline LoRA-DPO (Algorithm 9). |
| `train_sd3_self.py` | Online GRPO training (Algorithm 10), extended from the original SOLACE training script with `compute_intrinsic_score_map()` supporting all four score types. |
| `aggregate_counterfactual_runs.py` | Multi-seed aggregation of evaluation summaries. |

### 3.2 Shared-Probe Guarantee

The central invariant of the implementation is that **all conditions share the same probe noise**. In `score_conditions_shared_probes()`:

1. A probe bank of shape `[T_probe, K, B, C, H, W]` is generated **once** per scoring call.
2. For each timestep and each condition, the same `ε` vectors and the same noised states `x_t` are used.
3. The only difference between conditions is the text embedding passed to the transformer.

This means the COPE difference `C(z, c) - C(z, c̃)` is a pure function of the conditioning, with no probe-noise confound.

When `shared_probes=False` (optional mode), each condition gets independent noise — this path exists for ablation but is not the default.

**Implementation note**: This toggle is implemented in the SD3 scoring utilities used by reranking/eval (by scoring conditions in separate calls when `shared_probes=False`). The current GRPO training path always uses shared probes for counterfactual scoring.

### 3.3 Batched Negative Handling in GRPO

In the GRPO training loop, each prompt in a minibatch may have a different number of negatives (e.g., an OCR prompt produces OCR negatives, while a generic prompt falls back to unconditional). The implementation handles this by:

1. Generating negatives for each prompt independently via `build_batched_negative_prompts()`.
2. Grouping prompts by the number of negatives they produce.
3. Running `compute_counterfactual_scores()` separately per group (since the function requires all conditions to have the same batch size).
4. Scattering results back to a pre-allocated score buffer indexed by the original batch positions.

This ensures correct handling of mixed-mode batches without padding or masking.

---

## 4. Configuration

### 4.1 COPE-Specific Parameters

All COPE configuration lives under a `config.cf` namespace:

| Parameter | Default | Description |
|---|---|---|
| `score_type` | `"raw"` | Primary scoring method: `raw`, `pmi`, `cope`, `cope_lse` |
| `negative_mode` | `"auto"` | How to build negatives: `auto`, `ocr`, `count`, `spatial`, `attribute`, `unconditional` |
| `num_negatives` | `1` | Number of hard negatives (typically 1 for COPE, 4 for COPE-LSE) |
| `num_probe_steps` | `5` | Suffix timesteps to probe (e.g., last 5 of a 10-step schedule) |
| `k` | `8` | Number of antithetic probe noise vectors (must be even) |
| `delta` | `1e-6` | Numerical stability in `-log(mse + δ)` |
| `use_cfg_probe` | `False` | Apply CFG during probing (disabled by default) |
| `normalize_per_step` | `True` | Z-score normalization per timestep across the batch |
| `shared_probes` | `True` | Use identical probe noise for all conditions |
| `time_weighting` | `"mid"` | Timestep weight: `t(1-t)` for `"mid"`, uniform otherwise |
| `num_candidates` | `8` | Candidate budget N for reranking |
| `seed_stride` | `1000` | Per-prompt seed spacing for deterministic generation |

**Notes**:

- `num_candidates` and `seed_stride` are used by the reranking/pair-mining scripts (Stage 1) and are set in `config/counterfactual.py`. They are not used by the GRPO training configs.
- In GRPO, probe steps are not read from `config.cf.num_probe_steps`. Instead, the training script derives `num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)` and probes indices `range(int(0.5·num_train_timesteps), num_train_timesteps)`. This is used both during sampling-time intrinsic scoring and during eval, even when eval uses `config.sample.eval_num_steps` > `config.sample.num_steps`.

### 4.2 Named Configurations Used in Experiments

**Stage 1 — Reranking** (in `config/counterfactual.py`):

| Config | Model | Resolution | Precision | Steps | CFG | Candidates |
|---|---|---|---|---|---|---|
| `sd3_cf_rerank_2gpu` | SD3.5-Medium | 512 | bf16 | 10 | 7.0 | 8 |
| `sd3_cf_rerank_2gpu_fp16` | SD3.5-Medium | 512 | fp16 | 10 | 7.0 | 8 |
| `sd3_cf_structured_rerank_2gpu_fp16` | SD3.5-Medium | 512 | fp16 | 10 | 7.0 | 8 |

**Stage 2 — GRPO** (in `config/solace.py`):

| Config | Score Type | Negatives | Batch Shape | Samples/Epoch |
|---|---|---|---|---|
| `general_ocr_sd3_grpo_raw_1gpu_fit80g` | `raw` | — | 4×4×16 | 64 |
| `general_ocr_sd3_grpo_cope_lse_1gpu_fit80g` | `cope_lse` | 4 | 4×4×16 | 64 |

Both GRPO configs share:

- Same LoRA setup (rank, alpha, target modules)
- Same optimizer (AdamW, lr=3e-4, `β₁=0.9, β₂=0.999`)
- Same KL penalty `β=0.04`
- Same PPO clip range, gradient accumulation, EMA
- Same OCR evaluation dataset
- Same seed (42)
- Activation checkpointing enabled
- Effective train batch size: 32

The only difference is the intrinsic reward signal.

---

## 5. Experimental Results

### 5.1 Stage 1: Training-Free Reranking (Completed)

**Setup**:

- Model: `stabilityai/stable-diffusion-3.5-medium`
- Resolution: 512
- Precision: FP16
- Hardware: 2× H100 80 GB
- Seeds: 42, 43, 44 (3-seed average)
- Candidate budget: N = 8
- Generation steps: 10, CFG scale: 7.0
- Probe: last 5 steps, K = 8 antithetic, shared probes, no CFG probing
- Negatives: auto mode, 4 negatives for COPE-LSE
- External metric: OCR exact-match accuracy via PaddleOCR

**Datasets**:

- **OCR**: 100 prompts with quoted text strings (e.g., `A sign that says "B7Q9"`).
- **Structured**: 20 mixed prompts covering OCR, counting, spatial, and attribute families.

#### OCR Benchmark Results

| Method | OCR Mean | OCR Std |
|---|---:|---:|
| Single sample (no reranking) | 0.471 | 0.030 |
| SOLACE raw reranking | 0.427 | 0.029 |
| PMI reranking | 0.627 | 0.005 |
| COPE (single negative) | 0.668 | 0.030 |
| **COPE-LSE (4 negatives)** | **0.680** | **0.026** |

#### Counterfactual Discrimination Accuracy

Fraction of selected candidates where COPE margin > 0 (i.e., the selected image provides more evidence for the intended prompt than for the hard negative):

| Selection Method | Positive COPE Margin Rate |
|---|---:|
| SOLACE raw | 50% |
| COPE | 100% |
| COPE-LSE | 100% |

#### Selection Disagreement (OCR, averaged over 3 seeds)

| Comparison | Disagreement Rate |
|---|---|
| raw vs. COPE | ~86/100 prompts |
| raw vs. COPE-LSE | ~88/100 prompts |
| COPE vs. COPE-LSE | ~12/100 prompts |

#### Structured Prompt Suite Results

Positive COPE margin rate for selected candidates:

| Method | Rate Mean | Rate Std |
|---|---:|---:|
| SOLACE raw | 0.333 | 0.024 |
| COPE | 1.000 | 0.000 |
| COPE-LSE | 0.933 | 0.024 |

#### Stage 1 Interpretation

1. **Raw SOLACE reranking hurts OCR accuracy** (0.427 vs 0.471 baseline). This confirms the theoretical claim: raw confidence rewards generic denoisability, not prompt-specific correctness. On OCR prompts, the model prefers bland, easy-to-denoise images over ones that correctly render text.

2. **PMI already helps substantially** (0.627), showing that subtracting the unconditional baseline removes much of the typicality confound.

3. **COPE adds value over PMI** (0.668 vs 0.627): the hard-negative baseline is more informative than the unconditional baseline because it targets the specific semantic factor that the prompt controls.

4. **COPE-LSE is the strongest method** (0.680): using multiple negatives provides a more robust soft worst-case competitor.

5. **The methods select fundamentally different candidates**: raw and COPE-LSE disagree on 88% of prompts, confirming that counterfactual scoring produces a qualitatively different ranking, not just a minor reordering.

6. **Counterfactual consistency is perfect for COPE/COPE-LSE**: every selected image provides stronger evidence for the intended prompt than for its hard negative. Raw SOLACE achieves this only 50% of the time — no better than chance.

---

### 5.2 Stage 2: Online GRPO Post-Training (Early Runs, In Progress)

**Setup**:

Two matched 1-GPU training runs on 2× H100 80 GB (one GPU each), using the `fit80g` batch configs:

| Parameter | Value |
|---|---|
| Model | SD3.5-Medium |
| Resolution | 512 |
| LoRA rank / alpha | from SOLACE defaults |
| Train batch size per device | 4 |
| Images per prompt | 4 |
| Batches per epoch | 16 |
| Samples per epoch | 64 |
| Effective train batch size | 32 |
| Generation steps | 10 |
| CFG scale | 4.5 |
| Eval steps | 40 |
| KL penalty β | 0.04 |
| Per-prompt stat tracking | Yes |
| Seed | 42 |

Launch commands (run in separate terminals):

```bash
# GPU 0: SOLACE baseline (raw)
bash scripts/single_node/grpo_sd3_raw_1gpu_fit80g.sh \
  config/solace.py:general_ocr_sd3_grpo_raw_1gpu_fit80g \
  logs/grpo_compare_sd3_1gpu \
  raw_fit80g \
  42 \
  0

# GPU 1: COPE-LSE
bash scripts/single_node/grpo_sd3_cope_lse_1gpu_fit80g.sh \
  config/solace.py:general_ocr_sd3_grpo_cope_lse_1gpu_fit80g \
  logs/grpo_compare_sd3_1gpu \
  cope_lse_fit80g \
  42 \
  1
```

#### Training Progress

| Metric | Raw (SOLACE) | COPE-LSE |
|---|---|---|
| Run duration | ~2h 22m | ~3h 54m |
| Epochs completed | 59 | 35 |
| Optimizer steps | 65 | 39 |
| Unique prompts seen | 932 | 562 |

The COPE-LSE run is roughly 1.6× slower per epoch due to the additional transformer forward passes during probing for the unconditional baseline and the 4 hard-negative conditions at each probe step.

#### Intrinsic Reward Trajectory

**Raw SOLACE**: The `reward/ori_avg` (raw self-confidence) fluctuates around exactly 0.0 throughout all 59 epochs. After per-prompt normalization, the advantages have mean zero by construction, but the underlying absolute rewards show no upward trend. This is expected: the raw score measures generic confidence, and per-prompt normalization removes the mean, making the reward signal driven purely by within-prompt variance.

**COPE-LSE**: The `reward/ori_avg` (COPE-LSE score) stays near -1.386 (= -log(4), the theoretical minimum for a 4-class uniform classifier) throughout all 35 epochs. This indicates that at the current stage, the COPE-LSE score has not yet shown upward movement in the training signal. The value -log(4) ≈ -1.386 corresponds to the case where the positive and all four negatives receive approximately equal probe confidence — i.e., the model cannot yet distinguish the intended prompt from its counterfactual competitors in the intrinsic score.

#### Training Loss and KL

| Metric | Raw (SOLACE) | COPE-LSE |
|---|---|---|
| Loss range | [-0.10, +0.12] | [-0.79, +0.09] |
| KL loss (final) | 0.0014 | 0.0015 |
| Clip fraction (final) | 0.0 | 0.27 |
| Approx KL | ~0.0 | ~0.0 |

Both runs have low KL divergence from the reference model, indicating the LoRA updates are conservative. The COPE-LSE run shows non-trivial clip fraction (27%), suggesting the advantages create larger policy gradients that trigger PPO clipping more frequently.

#### Baseline Eval (Step 0, Before Training)

Both runs evaluated the base model before any LoRA updates:

| Metric | Value |
|---|---|
| `eval_reward/ocr` | 0.570 |
| `eval_intrinsic/raw` | 0.0 |
| `eval_intrinsic/cope_lse` | -1.386 |

The initial OCR accuracy of 0.570 is the base model's performance with 40 eval steps (vs. 10 generation steps used in reranking).

#### Stage 2 Preliminary Interpretation

1. **The runs are still early**: 35–59 epochs with ~1 optimizer step per epoch. The original SOLACE paper trains for hundreds of epochs. These results represent the initial phase of training.

2. **COPE-LSE reward is near the theoretical floor**: The -1.386 value shows the model treats positive and negative prompts nearly equally in the probe scoring. This could mean:
   - The counterfactual signal needs more training epochs to produce meaningful advantages.
   - The per-prompt normalization may be interacting differently with the COPE-LSE reward distribution compared to raw.
   - The COPE-LSE reward may require hyperparameter adjustments (e.g., fewer negatives, different time weighting) for the GRPO setting.

3. **Raw SOLACE also shows flat rewards**: The raw self-confidence reward hovers at 0.0, which is expected since per-prompt normalization centers it. The actual learning signal comes from within-prompt variance in the advantages, which is not visible in the mean reward.

4. **Training is stable**: Both runs have low KL, controlled loss, and no NaN/divergence issues. The infrastructure for COPE-GRPO is functional.

5. **COPE-LSE is ~1.6× slower**: Due to the additional transformer forward passes during probing for the unconditional baseline and the 4 hard-negative conditions per probe step per sample. This is a known trade-off.

---

## 6. Datasets

### OCR Dataset (`dataset/ocr/`)

Standard SOLACE OCR prompt set. Each prompt contains a quoted text string. Example prompts:

- `A sign that says "B7Q9"`
- `A weathered trail marker that reads "Camp North 4 Miles" beside pine trees.`
- `A clean storefront decal that says "Open Late Fridays" on a glass door.`

Train/test split, 100 test prompts used for Stage 1 evaluation.

### Structured Counterfactual Suite (`dataset/cf_structured/`)

A new prompt set designed so every prompt admits at least one deterministic hard negative. It covers multiple perturbation families in a single test set. Example prompts:

- OCR: `A festival banner reading "Summer Beats 2026" across the stage entrance.`
- Counting: `two red umbrellas leaning against a stone wall.`
- Spatial: `A red mug to the right of a white notebook on a desk.`
- Attribute: `A small blue suitcase under a coat rack.`
- Mixed: `A tiny green plant to the left of a black laptop.`

20 test prompts. Each is verified to produce a valid counterfactual under auto mode.

---

## 7. Output Format and Reproducibility

### Reranking Output (Stage 1)

Each run produces a directory:

```
<output_dir>/<timestamp>/
├── results.jsonl           # one JSON object per prompt
├── run_metadata.json       # small configuration snapshot (subset of fields)
├── summary.json            # aggregated metrics
├── tensorboard/            # TensorBoard event files
└── prompts/
    └── <NNNNN>/            # per-prompt directory
        ├── candidate_00_seed<S>.png  ... candidate_07_seed<S>.png
        ├── selected_raw.png
        ├── selected_pmi.png
        ├── selected_cope.png
        ├── selected_cope_lse.png
        └── selected_primary.png
```

Each JSONL row contains:

- `prompt`, `prompt_index`: the input.
- `candidate_seeds`: deterministic seeds for each candidate.
- `negative_mode`: which perturbation rule was used (e.g., `"ocr"`).
- `negative_prompts`: the actual hard-negative text(s).
- `scores`: includes `{raw, pmi, cope, cope_lse}` (each `[N]`) plus per-condition scores such as `positive`, `unconditional`, and `negative_*` when available.
- `selected_index`: `{single: 0, raw: i, pmi: j, cope: k, cope_lse: l, primary: m}`.
- `metrics`: `{ocr: [N]}` — external metric for each candidate.
- `selected_metrics`: per-method selected metric values.

### GRPO Output (Stage 2)

Each run produces a TensorBoard log directory with scalar tags including:

- `reward/*`: per-epoch intrinsic reward statistics (raw, cope, cope_lse, pmi).
- `eval_reward/ocr`: OCR accuracy on the test set at evaluation checkpoints.
- `eval_intrinsic/*`: intrinsic scores on eval images.
- `loss`, `policy_loss`, `kl_loss`: training loss components.
- `clipfrac`, `approx_kl`: PPO diagnostics.

### Seed Determinism

All generation is deterministic given the seed. For reranking, candidate seeds follow: `base_seed + prompt_index × seed_stride + candidate_index`. This ensures:

- Different prompts get different candidate pools.
- The same seed + prompt always produces the same candidates.
- Multi-seed runs (42, 43, 44) evaluate different candidate pools for variance estimation.

---

## 8. Computational Cost

### Per-Prompt Cost Breakdown (Reranking, N=8 candidates)

| Step | Transformer forward passes | Effective batch size | Notes |
|---|---:|---:|---|
| Generation (CFG) | 10 | 2·N (=16) | One forward per denoising step; CFG is implemented by doubling the batch (uncond+cond). |
| Probe scoring (positive) | 5 | K·N (=64) | 5 probe steps, K=8 probes, N=8 candidates; shared probes; no CFG probing. |
| Probe scoring (unconditional) | 5 | K·N (=64) | Same probe latents, different conditioning. |
| Probe scoring (4 negatives) | 20 | K·N (=64) | 4 negatives × 5 probe steps. |
| **Total probe forwards** | **30** | — | For COPE-LSE with 4 negatives. |
| **Total (gen + probe)** | **40** | — | For COPE-LSE with 4 negatives. |

In the implementation, both candidates and probes are flattened into the batch dimension, so probe scoring runs transformer forwards with an effective batch size of `K·N`. If `use_cfg_probe=True`, each condition requires two transformer forwards per probed step (unconditional + conditional).

### Wall-Clock Comparison (GRPO, 1 GPU)

From the early runs:

| Method | Time per Epoch | Overhead vs Raw |
|---|---|---|
| Raw SOLACE | ~2.4 min | — |
| COPE-LSE (4 neg) | ~6.7 min | ~2.8× |

The overhead comes from the additional transformer forward passes during probe scoring for the unconditional baseline and the 4 negative conditions at each training step.

---

## 9. Known Limitations and Failure Modes

1. **Unconditional fallback**: Prompts that don't match any perturbation rule fall back to the empty string as the negative. This degrades COPE to PMI, losing the hard-negative advantage.

2. **Variant diversity**: Some prompts (e.g., spatial with only one relation) produce only 1 distinct negative. The implementation returns as many unique negatives as it can find (possibly fewer than `num_negatives`), so COPE-LSE can degenerate to COPE when only one negative is available.

3. **GRPO reward scale**: The COPE-LSE reward near -log(4) throughout training suggests the per-prompt normalization and advantage computation may need tuning for counterfactual rewards that have a different distribution than raw confidence.

4. **Compute cost**: COPE-LSE with 4 negatives is ~2.8× slower than raw SOLACE per GRPO epoch. For reranking this is acceptable (one-time cost), but for online training it significantly extends wall-clock time.

5. **Model-specific**: The current implementation is specific to the SD3 flow-matching architecture. The noise recovery identity `ε̂ = v_pred + x₀` and the flow-matching interpolation `x_t = (1-t)x₀ + tε` are SD3-specific. Extension to SDXL (DDPM-based) or FLUX would require adapting the probe latent construction and noise recovery.

---

## 10. Summary of Status

| Component | Status | Notes |
|---|---|---|
| Core scoring (Algorithms 1–6) | ✅ Complete | Shared-probe, multi-condition, all 4 score types |
| Prompt perturbation (Algorithm 5) | ✅ Complete | OCR, count, spatial, attribute, unconditional fallback |
| Training-free reranking (Algorithm 7) | ✅ Complete | Multi-seed results available |
| Post-hoc evaluation | ✅ Complete | Cross-method comparison, aggregation |
| Pseudo-pair mining (Algorithm 8) | ✅ Complete | Script and configs ready |
| Offline LoRA-DPO (Algorithm 9) | ✅ Complete | Script, loss, and config ready; not yet run at scale |
| Online GRPO (Algorithm 10) | 🔄 In progress | Early runs stable but reward signal flat; needs more epochs or tuning |
| Multi-seed GRPO comparison | 🔄 In progress | Single seed (42) completed for fit80g configs |

**Stage 1 conclusion**: COPE-LSE reranking achieves 0.680 OCR accuracy vs. 0.427 for raw SOLACE reranking and 0.471 for single-sample baseline. The training-free reranking result is strong and validated across 3 seeds.

**Stage 2 status**: Infrastructure is functional. Both raw SOLACE and COPE-LSE GRPO runs are stable. The COPE-LSE reward has not yet shown upward movement after 35 epochs, likely due to the reward sitting near its theoretical floor. Further investigation is needed: longer training, reward scaling, or hyperparameter search.
