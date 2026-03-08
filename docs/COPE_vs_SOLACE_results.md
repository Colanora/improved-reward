# COPE vs SOLACE Results

This note summarizes the current multi-seed comparison between the SOLACE-style raw intrinsic confidence baseline and the COPE family on the SD3.5-Medium FP16 reranking pipeline.

Source reports:

- `logs/cf_vs_solace_complete_fp16/ocr_aggregate.json`
- `logs/cf_vs_solace_complete_fp16/structured_aggregate.json`

## Setup

- Model: `stabilityai/stable-diffusion-3.5-medium`
- Mode: training-free reranking
- Precision: FP16
- Hardware: 2x H100 80GB
- Seeds: `42, 43, 44`
- Candidate budget: `N=8`

Compared methods:

- `raw`: SOLACE-style raw intrinsic confidence
- `pmi`: unconditional baseline
- `cope`: single-negative counterfactual evidence
- `cope_lse`: multi-negative counterfactual evidence

## OCR Benchmark

Mean OCR score over 3 seeds:

| Method | OCR Mean | OCR Std |
|---|---:|---:|
| Single sample | 0.471 | 0.030 |
| SOLACE raw | 0.427 | 0.029 |
| PMI | 0.627 | 0.005 |
| COPE | 0.668 | 0.030 |
| COPE-LSE | 0.680 | 0.026 |

Main takeaways:

- `COPE-LSE` is the best OCR method in the current study.
- `COPE` also strongly outperforms the SOLACE raw baseline.
- `raw` performs worse than the single-sample baseline, which supports the claim that absolute intrinsic confidence is confounded by generic denoisability.

Selection behavior:

- `raw` vs `cope` disagree on about `86/100` prompts on average.
- `raw` vs `cope_lse` disagree on about `88/100` prompts on average.
- `cope` vs `cope_lse` still disagree on about `12/100` prompts, so multi-negative COPE is not collapsing to the single-negative case.

Counterfactual consistency:

- `raw` selected candidates have positive COPE margin on only `50%` of OCR prompts.
- `COPE` selected candidates are positive on `100%`.
- `COPE-LSE` selected candidates are positive on `100%`.

## Structured Counterfactual Suite

This suite currently gives an internal discrimination result rather than an external task metric.

Positive COPE margin rate:

| Method | Rate Mean | Rate Std |
|---|---:|---:|
| SOLACE raw | 0.333 | 0.024 |
| COPE | 1.000 | 0.000 |
| COPE-LSE | 0.933 | 0.024 |

Selection behavior:

- `raw` vs `cope` disagree on about `18.7/20` prompts.
- `raw` vs `cope_lse` disagree on about `18.3/20` prompts.
- `cope` vs `cope_lse` disagree on about `4/20` prompts.

Main takeaway:

- On structured prompts, COPE and COPE-LSE select candidates that are much more consistent with positive counterfactual evidence than the raw baseline.

## Interpretation

These results support the intended COPE story:

1. Raw intrinsic confidence is not a reliable alignment signal by itself.
2. Counterfactual intrinsic evidence is substantially more predictive of prompt correctness on OCR.
3. Multi-negative COPE-LSE is currently the strongest variant and should be treated as the main method.

## Recommended Next Steps

1. Use `COPE-LSE` as the primary method in follow-up experiments.
2. Keep `raw` as the SOLACE baseline and `pmi` as the intermediate ablation.
3. Inspect prompt-level disagreements between `raw` and `cope_lse` qualitatively.
4. Mine pseudo-preference pairs with `COPE-LSE`.
5. Add an external correctness metric or manual evaluation for the structured suite.
