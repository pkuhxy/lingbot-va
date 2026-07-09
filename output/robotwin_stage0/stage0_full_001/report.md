# Stage 0 Diagnostic Report

## Checkpoints and data

Run: `stage0_full_001`

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
| clean | linear_cka | 0.927479 |
| contrastive | linear_cka | 0.915515 |

## Action sensitivity

| checkpoint | metric | mean |
|---|---|---:|
| base | action_relative_l2 | 0.0820368 |
| base | action_relative_l2/first_25pct | 0 |
| base | action_relative_l2/last_75pct | 0.0876907 |
| clean | action_relative_l2 | 0.0192896 |
| clean | action_relative_l2/first_25pct | 0 |
| clean | action_relative_l2/last_75pct | 0.0199679 |
| contrastive | action_relative_l2 | 0.011794 |
| contrastive | action_relative_l2/first_25pct | 0 |
| contrastive | action_relative_l2/last_75pct | 0.0122415 |

## Future-video dynamics stability

| checkpoint | metric | mean |
|---|---|---:|
| base | latent_residual_cosine | 0.186414 |
| clean | latent_residual_cosine | 0.0813632 |
| contrastive | latent_residual_cosine | 0.0726447 |

## Directional finite difference

| checkpoint | metric | mean |
|---|---|---:|
| base | centered_directional_fd | 0.0755977 |
| base | directional_selectivity | 1.51195e+11 |
| clean | centered_directional_fd | 0.0537975 |
| clean | directional_selectivity | 1.07595e+11 |
| contrastive | centered_directional_fd | 0.0539623 |
| contrastive | directional_selectivity | 1.07925e+11 |

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
| contrastive | delta_vs_base/knn_k1 | -0.101442 |


## Joint diagnosis

- Both fine-tuned checkpoints substantially improve policy-path stability over base. Mean action relative L2 falls from 0.08204 (base) to 0.01929 (clean, -76.5%) and 0.01179 (contrastive, -85.6%).
- Contrastive is the strongest offline robustness candidate: its action sensitivity is 38.9% lower than clean, and its future-video residual instability is 10.7% lower than clean (0.07264 vs 0.08136).
- Directional finite differences are essentially tied for clean and contrastive (0.05380 vs 0.05396). This does not establish better geometry/3D equivariance for contrastive; the observed advantage is primarily appearance/action-path invariance.
- CKA remains high for both fine-tuned checkpoints (clean 0.9275, contrastive 0.9155). Together with improved action/video stability and no AUROC or balanced-accuracy domain-gap increase, there is no evidence here that fine-tuning destroyed the pretrained world representation.
- The clean-C2R kNN gap improves relative to base for both checkpoints, but clean improves more than contrastive (-0.1246 vs -0.1014). Domain-gap evidence therefore does not independently favor contrastive.
- Closed-loop coverage completed all 50 easy tasks and at least 49 of 50 background tasks, but this rollout evaluates only the clean checkpoint_step_5000. It is a behavioral sanity check, not a matched base/clean/contrastive comparison.
- Stage-0 decision: advance contrastive as the preferred offline candidate, while limiting the claim to improved nuisance/photometric robustness. Do not claim superior closed-loop task success or 3D generalization without a matched rollout subset across checkpoints.

## Limitations

Policy-path action sensitivity used the reduced 10-sample probe, so uncertainty is larger than for the 50-sample train-forward metrics. Reported action/video means retain identity rows. Latent residual dynamics remain VAE-dependent. The rollout is clean-checkpoint-only and cannot measure the causal effect of clean versus contrastive training.
