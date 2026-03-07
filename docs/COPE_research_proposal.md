# COPE: Counterfactual Prompt Evidence for Intrinsic Text-to-Image Alignment

## A self-contained research proposal, experiment plan, and code blueprint built on the SOLACE codebase

---

## 0. Executive Summary

This proposal develops a follow-up to **SOLACE** (“Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards”) that is:

1. **principled** rather than heuristic,
2. **simple** rather than adaptive or multi-component,
3. **cheap to iterate** on ~2 H100 GPUs,
4. **self-contained** so implementation does not depend on outside papers or proprietary infrastructure.

### Core idea

SOLACE scores a generated latent by how well the model reconstructs probe noise under the *intended prompt*.
That score is useful, but it confounds:

- generic image **typicality / denoisability**, and
- prompt-specific **alignment evidence**.

Our proposal fixes that by replacing raw self-confidence with a **counterfactual evidence ratio**:

\[
R_{\text{COPE}}(z;c,\tilde c)
=
\log \frac{p_\theta(z\mid c)}{p_\theta(z\mid \tilde c)}
\approx
C(z,c)-C(z,\tilde c),
\]

where:

- \(z\) is the generated terminal latent,
- \(c\) is the intended prompt,
- \(\tilde c\) is a minimally perturbed prompt (hard negative), and
- \(C(z,q)\) is the SOLACE-style probe confidence under condition \(q\).

### One-sentence thesis

**A generated image is aligned not when the model is merely confident in it, but when the image provides stronger intrinsic evidence for the intended prompt than for a nearby semantic competitor.**

This leads to a clean, testable, and novel direction:

- **Training-free reranking** with counterfactual intrinsic evidence.
- **Optional offline LoRA-DPO** using pseudo-preference pairs induced by the same evidence.

The main paper can stand on **training-free reranking alone**.
The LoRA stage is optional.

---

## 1. Why this direction is worth doing

### 1.1 What SOLACE already gives us

The public SOLACE repo already provides a strong starting point:

- supported backbones: **SD3.5-M/L, FLUX.1-dev, SDXL, WAN 2.1**,
- training entry points:
  - `scripts/train_sd3_self.py`
  - `scripts/train_sd3_self_ext.py`
  - `scripts/train_flux_self.py`
  - `scripts/train_sdxl_self.py`
  - `scripts/train_wan2_1_self.py`
- config system in `config/base.py` and `config/solace.py`,
- reward wrappers in `solace/rewards.py`,
- prompt utilities in `solace/prompts.py`,
- patched diffusion/flow pipelines in `solace/diffusers_patch/`.

Most importantly, `scripts/train_sd3_self.py` already contains the exact function we need conceptually: a **probe-based self-confidence scalar** computed from terminal latents with antithetic probes and no CFG during probing.

### 1.2 Why not just repeat SOLACE

A pure repetition of SOLACE is too incremental.
The real open question is **what raw self-confidence actually measures**.

Our central claim is that raw SOLACE confidence is approximately a conditional log-evidence term \(\log p_\theta(z\mid c)\), which mixes:

- prompt-independent typicality,
- prompt-specific alignment.

That explains both:

- why SOLACE improves OCR / compositional metrics,
- and why it can still drift toward bland, safe, easy-to-denoise samples.

### 1.3 Why this proposal is not “A+B” or “make it adaptive”

This proposal is a **single-objective replacement**:

- **Old object:** raw confidence \(C(z,c)\)
- **New object:** counterfactual evidence ratio \(C(z,c)-C(z,\tilde c)\)

This is not a weighted blend of unrelated terms, and it does not require dynamic schedules or adaptive controllers.
It is a single log-evidence ratio with a direct probabilistic interpretation.

---

## 2. Problem Statement

Given a pretrained text-to-image model \(p_\theta(z\mid c)\), we want an intrinsic signal that rewards **prompt-specific correctness**, not generic sample denoisability.

### Goal

Design a model-native alignment signal that:

1. requires **no external reward model**,
2. is **cheap enough** for best-of-N reranking on image models,
3. can optionally bootstrap **offline preference tuning**,
4. has a clear theoretical interpretation.

### Desired behavior

For a sample generated under prompt \(c\):

- the score should be **high** if the sample is good evidence for \(c\),
- the score should be **low** if the sample is also compatible with a nearby wrong prompt \(\tilde c\),
- the score should avoid rewarding “generic model comfort” alone.

---

## 3. Mathematical Formulation

## 3.1 SOLACE-style raw confidence

Let:

- \(z\) be the terminal latent,
- \(q\) a conditioning prompt,
- \(t\in\mathcal T\) probe timesteps,
- \(\epsilon^{(m)}\) probe noises,
- \(\hat\epsilon_\theta(\cdot\mid q)\) the model’s predicted probe noise under condition \(q\).

Define the probe MSE at timestep \(t\):

\[
\operatorname{MSE}_{t}(z,q)
=
\frac{1}{K}\sum_{m=1}^{K}
\left\|
\hat\epsilon_\theta(z_t^{(m)}, t, q)-\epsilon^{(m)}
\right\|^2.
\]

Then SOLACE-style confidence is:

\[
C(z,q)
=
\frac{1}{\sum_{t\in\mathcal T} w_t}
\sum_{t\in\mathcal T} w_t
\big[-\log(\operatorname{MSE}_{t}(z,q)+\delta)\big].
\]

In the released SOLACE SD3 code, the implemented self-confidence probe:

- uses **antithetic** probe noises,
- evaluates only the **suffix** of the trajectory,
- probes **without CFG**,
- normalizes step scores across the batch,
- uses a simple flow-matching blend \(x_t=(1-t)x_0+t\epsilon\),
- defines \(\hat\epsilon\) from the model output as `eps_hat = v_pred + x0`.

This proposal reuses that exact probe construction.

---

## 3.2 Why raw confidence is not enough

Under the same Gaussian residual interpretation that motivates SOLACE’s `-log(MSE)` reward,
raw confidence behaves like a conditional log-evidence term:

\[
C(z,c) \approx \text{const} + \log p_\theta(z\mid c).
\]

But:

\[
\log p_\theta(z\mid c)
=
\log p_\theta(z)
+
\underbrace{\log \frac{p_\theta(z\mid c)}{p_\theta(z)}}_{\text{prompt-specific evidence}}
-
\log p(c).
\]

So raw confidence contains **two** effects:

1. **Typicality / model comfort**: \(\log p_\theta(z)\)
2. **Prompt evidence**: \(\log p_\theta(z\mid c)-\log p_\theta(z)\)

This is the confound we want to remove.

---

## 3.3 The proposed score: Counterfactual Prompt Evidence

Let \(\tilde c\) be a minimally perturbed prompt that differs from \(c\) in exactly one target semantic factor (text string, count, relation, color, attribute, etc.).

Define:

\[
R_{\text{COPE}}(z;c,\tilde c)
=
C(z,c)-C(z,\tilde c).
\]

Interpretation:

- If \(R_{\text{COPE}}\) is large, the sample is better evidence for \(c\) than for the hard negative \(\tilde c\).
- If \(R_{\text{COPE}}\) is small, the sample is ambiguous or misaligned.

### Optional multi-negative form

If we have multiple negatives \(\tilde c_1,\ldots,\tilde c_M\), define:

\[
R_{\text{COPE-LSE}}(z;c,\{\tilde c_j\})
=
C(z,c)-\log\sum_{j=1}^{M} \exp C(z,\tilde c_j).
\]

This behaves like a soft worst-case Bayes factor against the nearest competitor.

### Optional unconditional special case

If \(\tilde c=\varnothing\) (empty prompt), then

\[
R_{\text{PMI}}(z;c)=C(z,c)-C(z,\varnothing)
\]

is the mutual-information / PMI-style special case.

However, **this is not the main novelty**.
The key novelty is the **counterfactual prompt baseline**, not the unconditional baseline.

---

## 3.4 Local Gaussian proposition (the insight)

Assume a local latent approximation:

\[
z\mid c \sim \mathcal N(\mu_c, s_c^2 I).
\]

Let the probe channel be:

\[
y = \alpha z + \sigma \epsilon, \quad \epsilon\sim \mathcal N(0,I).
\]

Then the Bayes-optimal probe loss for a fixed sample has the form:

\[
\ell_{\alpha}(z,c)
=
A_{\alpha}(s_c)+B_{\alpha}(s_c)\,\|z-\mu_c\|^2,
\]

where

\[
A_{\alpha}(s)
=
\frac{\alpha^4 s^4}{(\alpha^2 s^2+\sigma^2)^2}d,
\qquad
B_{\alpha}(s)
=
\frac{\alpha^2\sigma^2}{(\alpha^2 s^2+\sigma^2)^2}.
\]

### Interpretation

- \(A_{\alpha}(s_c)\) is a **prompt ambiguity baseline**.
- \(B_{\alpha}(s_c)\|z-\mu_c\|^2\) is a **sample–prompt mismatch** term.

So raw confidence is high either because:

- the prompt is inherently easy / narrow, or
- the sample truly matches the prompt.

That is exactly the confound we want to remove.

### Equal-covariance cancellation

If two nearby prompts have similar local dispersion (
\(s_c\approx s_{\tilde c}=s\)), then:

\[
\ell_{\alpha}(z,c)-\ell_{\alpha}(z,\tilde c)
=
B_{\alpha}(s)
\Big(
\|z-\mu_c\|^2-
\|z-\mu_{\tilde c}\|^2
\Big).
\]

The ambiguity baseline cancels.
The score becomes a **prompt discrimination** term.

This is the main theoretical justification for COPE.

---

## 3.5 Main proposition for the paper

### Proposition 1 (raw confidence conflates two effects)

Under the residual-Gaussian interpretation of the probe loss,
raw self-confidence is proportional to a conditional log-evidence term and therefore mixes:

- generic sample typicality,
- prompt-specific evidence.

### Proposition 2 (counterfactual confidence deconfounds ambiguity)

Under a local equal-covariance Gaussian approximation,
the difference of raw confidences between the intended prompt and a minimally perturbed prompt cancels prompt-ambiguity terms and yields a prompt-discriminative signal.

### Practical corollary

To improve prompt alignment, intrinsic rewards should not maximize raw confidence alone; they should maximize **relative evidence for the intended prompt over its closest semantic competitor**.

---

## 4. Research Questions

### RQ1
Does COPE correlate more strongly than raw SOLACE confidence with prompt-specific correctness on structured T2I tasks?

### RQ2
Does COPE improve **training-free reranking** over raw confidence at the same candidate budget?

### RQ3
Do pseudo-preference pairs mined by COPE produce better offline LoRA-DPO than pairs mined by raw confidence?

### RQ4
When does COPE fail?
Specifically:

- weak negatives,
- overly broad prompts,
- semantic perturbations not visible in the image,
- rare prompts where counterfactuals are badly chosen.

---

## 5. Proposed Method

## 5.1 Counterfactual prompt generator

We use **deterministic, rule-based prompt perturbations**.
This keeps the method self-contained and reproducible.

### Prompt perturbation rules

#### OCR prompts
If a prompt contains a quoted string:

- replace the quoted content with a different same-length alphanumeric string.

Example:

- Positive: `A sign that says "B7Q9"`
- Negative: `A sign that says "M2X4"`

#### Counting prompts
If the prompt contains a count word or digit:

- perturb by \(+1\) or \(-1\) within allowed range.

Example:

- Positive: `three red balls`
- Negative: `four red balls`

#### Spatial prompts
If the prompt contains a relation from a closed set:

- left ↔ right
- above ↔ below
- in front of ↔ behind

#### Attribute prompts
If the prompt contains a controlled attribute token:

- color swap from a small fixed palette,
- size swap from a small fixed set,
- material swap from a small fixed set.

### Fallback rule

If no deterministic perturbation is available, use the unconditional baseline:

\[
\tilde c = \varnothing.
\]

---

## 5.2 Probe score implementation

For each generated latent \(z\), evaluate the exact SOLACE-style probe scorer under both conditions:

- intended prompt \(c\),
- counterfactual prompt \(\tilde c\).

Use the **same**:

- terminal latent \(z\),
- probe timesteps,
- antithetic noise probes,
- noised probe states,
- no-CFG scoring setup.

This ensures the difference is attributable only to the conditioning prompt.

---

## 5.3 Training-free reranking

Given a prompt \(c\), generate \(N\) candidate images from different seeds.
For each candidate latent \(z^{(i)}\), compute:

\[
R_i = R_{\text{COPE}}(z^{(i)};c,\tilde c).
\]

Choose the top candidate:

\[
i^\star = \arg\max_i R_i.
\]

This is the **core paper**.
No post-training is required.

---

## 5.4 Optional offline DPO

Generate candidate pairs \((z^+, z^-)\) under the same prompt \(c\).
Label the one with higher COPE score as preferred.
Then perform LoRA-only DPO or safeguarded DPO.

This stage is optional and should only be attempted after the reranking signal is validated.

---

## 6. What exactly we reuse from SOLACE

We will reuse the following from the existing repo:

### Existing repo facts we rely on

- The repo is public under MIT license.
- It supports SD3.5, FLUX, SDXL, WAN 2.1.
- `scripts/train_sd3_self.py` contains the current probe scorer and SD3 LoRA training loop.
- `config/solace.py` already contains:
  - SD3.5-M/L configs,
  - FLUX config,
  - SDXL configs,
  - WAN 2.1 video config.
- `solace/rewards.py` already provides OCR, PickScore, ImageReward, CLIP, Qwen-VL, GenEval wrappers.
- `solace/prompts.py` already has prompt-loading helpers and OCR prompt sources.

### Existing implementation behavior we intentionally preserve

1. **Probe only suffix timesteps**.
2. **Use antithetic probes**.
3. **Disable CFG during probing**.
4. **Compute score in latent space**.
5. **Use LoRA-only updates if training is enabled**.

---

## 7. Minimal novelty claim

The novelty claim should be narrow and defensible:

> Existing intrinsic T2I methods reward **conditional denoising confidence**. We show that this quantity conflates generic typicality with prompt-specific evidence. We propose a counterfactual Bayes-factor intrinsic score that measures whether a generated latent is better supported by the intended prompt than by a minimal semantic competitor. This yields a simpler and more faithful alignment signal for training-free reranking and lightweight post-training.

Avoid over-claiming.
Do **not** claim the first use of MI-like quantities in diffusion.
Do **not** claim the first use of prompt perturbations in T2I alignment.
The genuine combination is:

- intrinsic probe score,
- counterfactual prompt baseline,
- sample-level Bayes-factor view,
- training-free reranking as the main result.

---

## 8. Experimental Plan

## 8.1 Scope

### Primary scope (must-do)

- **Model:** SD3.5-Medium
- **Resolution:** 512
- **Mode:** training-free reranking only
- **Tasks:** OCR + structured synthetic prompts

### Secondary scope (if time permits)

- **Transfer test:** SDXL or FLUX.1-dev
- **Mode:** training-free reranking

### Optional scope (only after signal works)

- **LoRA-only offline DPO** on SD3.5-Medium

### Out of scope initially

- Online GRPO / RL reproduction
- Video
- Large-scale human study

---

## 8.2 Datasets

### Dataset A: OCR

Use the SOLACE OCR dataset format:

- `dataset/ocr/train.txt`
- `dataset/ocr/test.txt`

Metric:

- OCR exact-match accuracy using `solace.rewards.ocr_score()`.

### Dataset B: Structured synthetic prompt suite (new)

Create a new folder:

- `dataset/cf_structured/train.txt`
- `dataset/cf_structured/test.txt`
- optional metadata JSONL

Include prompt families:

1. OCR text rendering
2. counting
3. color attributes
4. spatial relations
5. size/attribute swaps

Each prompt should admit a deterministic hard negative.

### Optional Dataset C: GenEval

If the existing local GenEval server is available, run it as an optional benchmark using the repo’s existing `geneval_score` wrapper.
This is not required for the core paper.

---

## 8.3 Metrics

### Primary metrics

1. **OCR accuracy**
2. **Counterfactual discrimination accuracy**
   - For a generated image from prompt \(c\), does the intrinsic score satisfy
     \(R(z;c,\tilde c) > 0\)?
   - For multiple negatives, is the positive prompt ranked highest?
3. **Best-of-N improvement curve**
   - improvement in OCR / structured prompt accuracy vs candidate budget \(N\)

### Secondary metrics

4. CLIPScore
5. PickScore
6. ImageReward

### Diagnostic metrics

7. correlation between score and OCR success
8. score margin:
   \(C(z,c)-C(z,\tilde c)\)
9. diversity / collapse diagnostics:
   - mean pairwise CLIP image similarity among top-ranked images,
   - prompt-conditioned latent variance,
   - simple image entropy measures.

---

## 8.4 Baselines

### Baseline B0 — single sample
Generate 1 image and keep it.

### Baseline B1 — best-of-N by raw SOLACE confidence
\[
\arg\max_i C(z^{(i)},c)
\]

### Baseline B2 — best-of-N by unconditional PMI-style score
\[
\arg\max_i \big(C(z^{(i)},c)-C(z^{(i)},\varnothing)\big)
\]

### Baseline B3 — best-of-N by single-negative COPE (ours)
\[
\arg\max_i \big(C(z^{(i)},c)-C(z^{(i)},\tilde c)\big)
\]

### Baseline B4 — best-of-N by multi-negative COPE-LSE (ours)
\[
\arg\max_i \Big(C(z^{(i)},c)-\log\sum_j e^{C(z^{(i)},\tilde c_j)}\Big)
\]

### Optional external baseline B5 — best-of-N by PickScore
Use only as an optional reference, not as the core comparison.

---

## 8.5 Hypotheses

### H1
COPE improves prompt-specific task metrics (OCR / structured alignment) over raw SOLACE reranking at the same candidate budget.

### H2
COPE is more robust to bland-collapse failure modes than raw SOLACE reranking because it subtracts generic denoisability.

### H3
Pseudo-pairs mined by COPE yield stronger offline LoRA-DPO than pairs mined by raw SOLACE confidence.

---

## 8.6 Minimal publication package

A NeurIPS poster paper can be supported by:

1. one strong theoretical proposition,
2. one clean training-free method,
3. one existing automatic metric (OCR),
4. one controlled synthetic benchmark,
5. one transfer model (optional),
6. one small offline DPO extension (optional).

The paper does **not** need online RL.

---

## 9. Compute Plan for ~2 H100 GPUs

## 9.1 Main recommendation

### Start with training-free reranking only

This stage requires:

- no model update,
- no optimizer tuning,
- no distributed RL instability,
- only generation + probe scoring.

### Recommended candidate settings

For fast iteration:

- model: `stabilityai/stable-diffusion-3.5-medium`
- resolution: 512
- generation steps: 10
- probe steps: last 5 generation timesteps
- probes: `K = 8` antithetic
- candidates: `N = 4` during debugging, `N = 8` for final runs

### GPU layout

- GPU 0: generation + probe scoring
- GPU 1: metric evaluation / next batch prefetch / optional VLM metrics

---

## 9.2 Optional LoRA-DPO stage

Only after reranking works.

Suggested safe setup:

- model: SD3.5-Medium
- LoRA only on transformer attention blocks
- rank 32, alpha 64
- batch size 2–4 per GPU
- gradient accumulation 4
- 5k–20k pseudo-pairs to start
- early stopping on OCR + validation score

This is far cheaper and more stable than online GRPO.

---

## 10. Codebase Design

## 10.1 Existing SOLACE tree we reuse

```text
SOLACE/
├── config/
│   ├── base.py
│   └── solace.py
├── dataset/
├── scripts/
│   ├── train_sd3_self.py
│   ├── train_sd3_self_ext.py
│   ├── train_flux_self.py
│   ├── train_sdxl_self.py
│   └── train_wan2_1_self.py
└── solace/
    ├── rewards.py
    ├── prompts.py
    ├── stat_tracking.py
    └── diffusers_patch/
```

## 10.2 Proposed new tree

```text
SOLACE/
├── config/
│   ├── base.py
│   ├── solace.py
│   └── counterfactual.py                  # NEW configs for COPE experiments
├── dataset/
│   ├── ocr/
│   └── cf_structured/                     # NEW structured prompts
├── scripts/
│   ├── train_sd3_self.py
│   ├── rerank_counterfactual_sd3.py       # NEW main experiment
│   ├── eval_counterfactual_sd3.py         # NEW evaluation-only script
│   ├── build_cf_pairs_sd3.py              # NEW pseudo-pair mining
│   └── train_sd3_cf_dpo.py                # NEW optional offline DPO
└── solace/
    ├── rewards.py
    ├── prompts.py
    ├── counterfactual_reward.py           # NEW core scorer
    ├── baseline_prompts.py                # NEW prompt perturbations
    ├── probe_utils.py                     # NEW refactor of probe code
    ├── dpo_utils.py                       # NEW optional
    └── diffusers_patch/
```

---

## 10.3 File-by-file implementation notes

### `solace/probe_utils.py` (new)
Purpose: factor out the probe code from `train_sd3_self.py`.

Functions:

- `build_antithetic_probes(K, shape, device, dtype)`
- `make_probe_latents(x0, t_idx, eps, mode="flow")`
- `score_condition(transformer, x0, timesteps, prompt_embeds, pooled_prompt_embeds, neg_prompt_embeds, neg_pooled_prompt_embeds, config, use_steps=None)`
- `score_conditions_shared_probes(transformer, x0, timesteps, cond_list, shared_eps, config, use_steps=None)`

### `solace/baseline_prompts.py` (new)
Purpose: deterministic construction of hard negatives.

Functions:

- `build_unconditional(prompt)`
- `build_ocr_negative(prompt)`
- `build_count_negative(prompt)`
- `build_spatial_negative(prompt)`
- `build_attribute_negative(prompt)`
- `build_counterfactuals(prompt, mode="auto", n_neg=1)`

### `solace/counterfactual_reward.py` (new)
Purpose: implement COPE.

Functions:

- `raw_confidence(...) -> Tensor[B]`
- `pmi_confidence(...) -> Tensor[B]`
- `counterfactual_confidence(...) -> Tensor[B]`
- `counterfactual_confidence_lse(...) -> Tensor[B]`

### `scripts/rerank_counterfactual_sd3.py` (new)
Purpose: main paper experiment.

Responsibilities:

1. load SD3.5-M pipeline,
2. generate `N` candidates per prompt,
3. compute B0/B1/B2/B3/B4 scores,
4. rerank,
5. save images + metrics,
6. dump per-sample JSON for analysis.

### `scripts/build_cf_pairs_sd3.py` (new)
Purpose: build pseudo-preference pairs from COPE scores.

Output format:

```json
{
  "prompt": "...",
  "winner_seed": 123,
  "loser_seed": 456,
  "winner_score": 1.23,
  "loser_score": -0.47,
  "winner_path": "...",
  "loser_path": "...",
  "negative_prompt": "..."
}
```

### `scripts/train_sd3_cf_dpo.py` (new)
Purpose: optional offline DPO / SDPO LoRA tuning.

Use only after training-free reranking is validated.

---

## 11. Pseudocode

## Algorithm 1: Counterfactual prompt generation

```text
Algorithm 1 BuildCounterfactualPrompt(prompt c)
Input: prompt c
Output: one or more hard-negative prompts {ĉ_j}

1: if c contains quoted text s then
2:     sample s' with same length and character class, s' != s
3:     return { replace quoted text s with s' }
4: else if c contains count token n then
5:     choose n' in {n-1, n+1} within valid range
6:     return { replace n with n' }
7: else if c contains a spatial relation r in {left, right, above, below, in front of, behind} then
8:     return { replace r with opposite(r) }
9: else if c contains an attribute token a in a controlled lexicon then
10:    return { replace a with a different token from same lexicon }
11: else
12:    return { empty prompt }
```

## Algorithm 2: Raw probe confidence (SOLACE-compatible)

```text
Algorithm 2 RawConfidence(z, c)
Input: terminal latent z, prompt c, probe timesteps T, K antithetic probes
Output: scalar confidence C(z,c)

1: Initialize scores S_t = 0 for each t in T
2: for each probe timestep t in suffix(T) do
3:     sample K/2 probe noises ε_1, ..., ε_{K/2}
4:     create antithetic set {ε_1, ..., ε_{K/2}, -ε_1, ..., -ε_{K/2}}
5:     for each ε_k do
6:         create noised latent z_t^{(k)} using the same probe channel as SOLACE
7:         predict ε̂_k = model(z_t^{(k)}, t, c) using no-CFG probing
8:         compute mse_k = ||ε̂_k - ε_k||^2
9:     end for
10:    mse_t = mean_k mse_k
11:    S_t = -log(mse_t + δ)
12: end for
13: normalize per-step scores if desired (same as SOLACE implementation)
14: return weighted mean of {S_t}
```

## Algorithm 3: Counterfactual Prompt Evidence (COPE)

```text
Algorithm 3 COPE(z, c)
Input: terminal latent z, intended prompt c
Output: counterfactual intrinsic score R_COPE(z;c)

1: construct one or more hard-negative prompts {ĉ_j} = BuildCounterfactualPrompt(c)
2: compute C_pos = RawConfidence(z, c)
3: if using a single negative then
4:     compute C_neg = RawConfidence(z, ĉ_1)
5:     return C_pos - C_neg
6: else
7:     for each ĉ_j compute C_j = RawConfidence(z, ĉ_j)
8:     return C_pos - logsumexp_j(C_j)
```

## Algorithm 4: Training-free reranking

```text
Algorithm 4 RerankByCOPE(prompt c, candidate count N)
Input: prompt c, candidate count N
Output: selected image x*

1: Generate N candidate trajectories under prompt c with seeds s_1, ..., s_N
2: Extract terminal latents z_1, ..., z_N
3: for i = 1 to N do
4:     score_i = COPE(z_i, c)
5: end for
6: choose i* = argmax_i score_i
7: decode and return image x_{i*}
```

## Algorithm 5: Pseudo-pair mining for offline DPO

```text
Algorithm 5 BuildPseudoPairs(D)
Input: prompt set D
Output: pseudo-preference dataset P

1: Initialize empty set P
2: for each prompt c in D do
3:     generate N candidates {z_i} under c
4:     compute scores r_i = COPE(z_i, c)
5:     choose winner w = argmax_i r_i
6:     choose loser l = argmin_i r_i   or a hard loser with small margin threshold
7:     if r_w - r_l >= τ then
8:         add (c, z_w, z_l, r_w, r_l) to P
9:     end if
10: end for
11: return P
```

## Algorithm 6: Optional offline LoRA-DPO

```text
Algorithm 6 TrainCOPE-DPO(model θ, pseudo-pairs P)
Input: pretrained T2I model θ, pseudo-pairs P
Output: LoRA-adapted model θ'

1: freeze base model weights
2: attach LoRA adapters to transformer attention blocks
3: for each minibatch of preference pairs (c, z+, z-) do
4:     compute diffusion/flow log-likelihood surrogate for z+ and z-
5:     compute DPO or safeguarded-DPO loss
6:     update only LoRA parameters
7: end for
8: early stop on validation OCR / COPE reranking improvement
9: return θ'
```

---

## 12. Main Script Designs

## 12.1 `scripts/rerank_counterfactual_sd3.py`

### Inputs

- `--config config/counterfactual.py:sd3_cf_rerank_2gpu`
- prompt dataset path
- candidate count `N`
- score type in `{raw, pmi, cope, cope_lse}`
- output directory

### Outputs

- per-prompt selected image
- full candidate images (optional)
- JSONL with:
  - prompt
  - candidate seeds
  - scores by method
  - selected index
  - negative prompt(s)
  - OCR / optional metrics

### Pseudocode sketch

```text
load pipeline
load dataset
for each prompt c in dataset:
    generate N candidate trajectories under c
    for each candidate latent z_i:
        build counterfactuals ĉ_j
        compute raw / PMI / COPE scores
    rerank candidates under each scoring rule
    decode selected candidates
    evaluate OCR and optional metrics
save results table and per-sample JSONL
```

---

## 12.2 `scripts/build_cf_pairs_sd3.py`

### Inputs

- prompt dataset
- candidate count `N`
- margin threshold `τ`
- scoring rule (`cope` or `cope_lse`)

### Outputs

- pseudo-pair dataset JSONL
- optional cached image directory

### Design decision

Store both:

- image paths,
- terminal latent paths if convenient,
- scores and prompts.

This keeps the DPO stage decoupled from pair mining.

---

## 12.3 `scripts/train_sd3_cf_dpo.py`

### Inputs

- pseudo-pair JSONL
- base model checkpoint
- LoRA config
- training hyperparameters

### Outputs

- LoRA checkpoint
- evaluation table

### Recommendation

If coding time is limited, use Diffusion-DPO-style objective first.
If quality regressions appear, switch to safeguarded DPO.

---

## 13. Recommended Configs

## 13.1 Training-free reranking config (main paper)

```python
# config/counterfactual.py

def sd3_cf_rerank_2gpu():
    config = base.get_config()
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    config.resolution = 512
    config.mixed_precision = "bf16"

    config.sample.num_steps = 10
    config.sample.eval_num_steps = 10
    config.sample.guidance_scale = 7.0

    config.cf = ml_collections.ConfigDict()
    config.cf.num_candidates = 8
    config.cf.num_probe_steps = 5       # last 5 of 10
    config.cf.k = 8                     # antithetic probes
    config.cf.delta = 1e-6
    config.cf.score_type = "cope"
    config.cf.negative_mode = "auto"
    config.cf.num_negatives = 1
    config.cf.use_cfg_probe = False

    config.output_dir = "logs/cf_rerank_sd3"
    return config
```

## 13.2 Optional LoRA-DPO config

```python

def sd3_cf_dpo_2gpu():
    config = sd3_cf_rerank_2gpu()
    config.train = ml_collections.ConfigDict()
    config.train.learning_rate = 1e-4
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 4
    config.train.max_steps = 3000
    config.train.eval_every = 200
    config.train.use_lora = True
    config.train.lora_rank = 32
    config.train.lora_alpha = 64
    config.train.lora_dropout = 0.0
    config.train.target_modules = [
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
        "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
    ]
    config.dpo = ml_collections.ConfigDict()
    config.dpo.beta = 0.1
    config.dpo.margin_threshold = 0.25
    config.output_dir = "logs/cf_dpo_sd3"
    return config
```

---

## 14. Ablation Matrix

### Core ablations

1. **Score type**
   - raw
   - PMI
   - COPE single negative
   - COPE-LSE multi-negative

2. **Negative type**
   - unconditional
   - OCR replacement
   - count perturbation
   - spatial swap
   - attribute swap

3. **Candidate budget**
   - N = 1, 2, 4, 8

4. **Probe budget**
   - K = 4, 8

5. **Probe timesteps**
   - last 3, last 5, last 7 steps

### Optional ablations

6. **shared vs non-shared probes across positive/negative conditions**
7. **single negative vs logsumexp of 2–4 negatives**
8. **with / without per-step normalization**

---

## 15. Sanity Checks and Failure Diagnostics

## 15.1 If COPE ≈ raw SOLACE

This means negatives are too weak or too generic.

Fix:

- strengthen perturbation rule,
- ensure only one semantic factor changes,
- use same-length OCR substitutions,
- use exact opposite relations.

## 15.2 If COPE is too noisy

This means probe variance dominates.

Fix:

- keep `K=8` antithetic probes,
- use shared probes across positive/negative conditions,
- restrict to suffix timesteps only.

## 15.3 If OCR improves but image quality drops

Fix:

- use reranking only for paper main result,
- keep DPO optional,
- if training, use smaller LoRA LR or safeguarded DPO.

## 15.4 If DPO hurts winner quality

Switch to safeguarded DPO.
Do not make DPO the main claim.

---

## 16. Why this is a NeurIPS-poster-worthy paper

This direction is poster-worthy if the final paper demonstrates three things clearly:

1. **A sharp conceptual diagnosis**
   - raw intrinsic confidence is not alignment; it is conditional evidence mixed with typicality.

2. **A minimal, principled fix**
   - replace raw confidence with a counterfactual evidence ratio.

3. **A cheap, strong use case**
   - training-free reranking improves structured prompt fidelity without external reward models.

This is enough for a convincing poster paper.
It does not require a giant training campaign.

---

## 17. Deliverables

## Deliverable D1 — Core theorem + analysis

- Proposition 1 (confound)
- Proposition 2 (counterfactual cancellation)
- proof sketch in appendix

## Deliverable D2 — Main experiment

- training-free reranking on OCR + structured prompts
- comparison of raw / PMI / COPE / COPE-LSE

## Deliverable D3 — Optional extension

- offline pseudo-pair LoRA-DPO

## Deliverable D4 — Open-source release

- code integrated into SOLACE-style repo
- configs for 2-GPU reranking and optional DPO
- JSONL outputs + plotting scripts

---

## 18. Concrete 6-Week Plan

### Week 1
- factor out probe scorer from `train_sd3_self.py`
- reproduce raw SOLACE score on SD3.5-M for OCR prompts
- verify antithetic probes and no-CFG scoring

### Week 2
- implement `baseline_prompts.py`
- add OCR negative generator
- implement COPE scoring for a single negative
- run small reranking study on 100 prompts

### Week 3
- add counting / spatial / attribute perturbations
- build structured prompt set
- produce main reranking tables for raw vs PMI vs COPE

### Week 4
- add multi-negative COPE-LSE
- run candidate-budget curves (N=1,2,4,8)
- run probe-budget curves (K=4,8)

### Week 5
- optional transfer to SDXL or FLUX
- polish theorem and write analysis section

### Week 6
- if time remains, mine pseudo-pairs and run one LoRA-DPO pilot
- finalize figures and appendix

---

## 19. Final Recommendation

If time or compute is tight, make the paper **only** this:

> **COPE reranks candidate images using an intrinsic counterfactual Bayes factor computed from the generator’s own self-denoising probes.**

That paper is:

- simple,
- principled,
- self-contained,
- easy to implement on top of SOLACE,
- not just “SOLACE but adaptive”.

The LoRA-DPO stage is optional.
The core contribution is the intrinsic score itself.

---

## 20. Minimal references to include in the paper

1. SOLACE / ARC — intrinsic self-confidence rewards for T2I diffusion.
2. Information Theoretic Text-to-Image Alignment (MI-TUNE) — unconditional MI / PMI alignment from the model itself.
3. RFMI — mutual-information estimation for rectified flow.
4. Interpretable Diffusion via Information Decomposition — diffusion and information identities.
5. Free Lunch Alignment / TPO — prompt perturbation without preference image pairs.
6. OSPO — object-centric self-improving preference optimization.
7. Diffusion-DPO — preference optimization for diffusion.
8. Diffusion-SDPO — safeguarded DPO for diffusion.
9. TTSnap / Probe-Select / Diffusion Probe — early internal signals for test-time selection.

---

## Final one-line paper pitch

**Counterfactual Prompt Evidence (COPE): generated images should be scored not by how confidently the model denoises them under the intended prompt, but by how much more confidently it denoises them under the intended prompt than under the nearest semantic competitor.**

