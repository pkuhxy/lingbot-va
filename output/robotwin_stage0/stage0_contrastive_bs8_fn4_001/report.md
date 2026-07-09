# Stage 0 Diagnostic Report

## Checkpoints and data

Run: `stage0_contrastive_bs8_fn4_001`

## Determinism and sanity checks

Checkpoint audit: **PASS**.

Identity rows are retained in action/video tables and must be checked before interpreting perturbations.

## Parameter update audit

- base: relative L2 `0.0`, nonfinite `0`
- clean: relative L2 `0.03221774715715278`, nonfinite `0`
- contrastive: relative L2 `0.036514413042589226`, nonfinite `0`

## Layer-wise feature drift

| checkpoint | metric | mean |
|---|---|---:|
| base | linear_cka | 1 |
| clean | linear_cka | 0.927479 |
| contrastive | linear_cka | 0.778039 |

## Action sensitivity

| checkpoint | metric | mean |
|---|---|---:|
| base | action_relative_l2 | 0.0570036 |
| base | action_relative_l2/first_25pct | 0 |
| base | action_relative_l2/last_75pct | 0.0605251 |
| clean | action_relative_l2 | 0.0178843 |
| clean | action_relative_l2/first_25pct | 0 |
| clean | action_relative_l2/last_75pct | 0.018528 |
| contrastive | action_relative_l2 | 0.0184803 |
| contrastive | action_relative_l2/first_25pct | 0 |
| contrastive | action_relative_l2/last_75pct | 0.0191526 |

## Future-video dynamics stability

| checkpoint | metric | mean |
|---|---|---:|
| base | latent_residual_cosine | 0.155445 |
| clean | latent_residual_cosine | 0.0763736 |
| contrastive | latent_residual_cosine | 0.109213 |

## Directional finite difference

| checkpoint | metric | mean |
|---|---|---:|
| base | centered_directional_fd | 0.0755977 |
| base | directional_selectivity | 1.51195e+11 |
| clean | centered_directional_fd | 0.0537975 |
| clean | directional_selectivity | 1.07595e+11 |
| contrastive | centered_directional_fd | 0.0530421 |
| contrastive | directional_selectivity | 1.06084e+11 |

## Clean-C2R domain gap

| checkpoint | metric | mean |
|---|---|---:|
| base | delta_vs_base/auroc | 0 |
| base | delta_vs_base/balanced_accuracy | 0 |
| base | delta_vs_base/knn_k1 | 0 |
| clean | delta_vs_base/auroc | 0 |
| clean | delta_vs_base/balanced_accuracy | 0 |
| clean | delta_vs_base/knn_k1 | -0.124637 |
| contrastive | delta_vs_base/auroc | 0 |
| contrastive | delta_vs_base/balanced_accuracy | 0 |
| contrastive | delta_vs_base/knn_k1 | -0.102696 |

## Joint diagnosis

Apply the fixed decision table from `STAGE0_TEST_IMPLEMENTATION_GUIDE.md`; do not infer policy quality from CKA or a single offline metric.

## Limitations

Latent residual dynamics remain VAE-dependent. Initial and final C2R observations are reported separately. Closed-loop rollout success remains the behavioral ground truth.
