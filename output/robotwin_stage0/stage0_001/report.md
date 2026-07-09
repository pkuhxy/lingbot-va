# Stage 0 Diagnostic Report

## Checkpoints and data

Run: `stage0_001`

## Determinism and sanity checks

Checkpoint audit: **PASS**.

Identity rows are retained in action/video tables and must be checked before interpreting perturbations.

## Parameter update audit

- base: relative L2 `0.0`, nonfinite `0`
- clean: relative L2 `0.03221774715715278`, nonfinite `0`
- contrastive: relative L2 `0.032694298243622445`, nonfinite `0`

## Layer-wise feature drift

| checkpoint | metric | mean |
|---|---|---:|
| base | linear_cka | 1 |
| clean | linear_cka | 0.951212 |
| contrastive | linear_cka | 0.939619 |

## Action sensitivity

| checkpoint | metric | mean |
|---|---|---:|
| base | action_relative_l2 | 0.0136851 |
| base | action_relative_l2/first_25pct | 0 |
| base | action_relative_l2/last_75pct | 0.0141708 |
| clean | action_relative_l2 | 0.0153228 |
| clean | action_relative_l2/first_25pct | 0 |
| clean | action_relative_l2/last_75pct | 0.0158544 |
| contrastive | action_relative_l2 | 0.0221856 |
| contrastive | action_relative_l2/first_25pct | 0 |
| contrastive | action_relative_l2/last_75pct | 0.0229595 |

## Future-video dynamics stability

| checkpoint | metric | mean |
|---|---|---:|
| base | latent_residual_cosine | 0.171222 |
| clean | latent_residual_cosine | 0.074282 |
| contrastive | latent_residual_cosine | 0.0891106 |

## Directional finite difference

No completed metric table was found.

## Clean-C2R domain gap

No completed metric table was found.

## Joint diagnosis

Apply the fixed decision table from `STAGE0_TEST_IMPLEMENTATION_GUIDE.md`; do not infer policy quality from CKA or a single offline metric.

## Limitations

Latent residual dynamics remain VAE-dependent. Initial and final C2R observations are reported separately. Closed-loop rollout success remains the behavioral ground truth.
