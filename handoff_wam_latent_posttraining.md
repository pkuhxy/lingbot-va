# WAM 表征后训练研究内容摘要

## 1. 研究目标

目标是研究一种针对 World Action Model, WAM 的后训练方法，用少量数据、不大改模型架构的方式，增强 WAM 内部 `video latent` 与 `action latent` 在表征层面的关系。

核心问题不是继续做仿真场景 SFT 或 benchmark 过拟合，而是让 WAM 的隐空间更好地表达：

- action latent 能准确表征需要执行的 action；
- video latent 不只是包含 pixel / 背景 / 图像细节，也包含 action 导致的状态转移信息；
- action latent 不应该和完整 video latent 硬对齐，而应该解释 video latent 中的 controllable / actor-centric dynamics 子空间；
- 后训练应提升 action-conditioned dynamics representation，而不是只提升某个任务成功率。

## 2. Baseline

计划以 LingBot-VA 的 SFT 后模型作为 baseline。

记为：

```text
LingBot-VA-SFT
```

它本身已经包含 video/world branch 与 action branch，因此适合作为 WAM 内部 video-action latent 表征后训练的起点。

## 3. 和 World2Act 的关系

World2Act 原本是做 World Model 和 VLA 之间的 latent-level 对齐：

```text
World Model video latent -> Video Adapter -> shared latent space
VLA/action latent        -> Action Adapter -> shared latent space
```

然后用 shared latent 去后训练 VLA residual policy。

迁移到 WAM 时，不再是两个外部模型之间的迁移，而是：

```text
WAM 内部 video/world branch latent
↔
WAM 内部 action branch latent
```

因此 adapter 的角色应从“跨模型桥接器”变成“训练时的表征探针/约束器”：

```text
Video Adapter: 从 video latent 中提取 controllable dynamics 子空间
Action Adapter: 从 action latent 中提取 action-causal 子空间
```

## 4. 对 World2Act 式对齐的 critique

直接把 video latent 和 action latent 对齐到同一个 shared space 并不完全合理，因为二者信息不对称：

```text
video latent = 当前状态感知 + 背景/物体/视角 + 未来状态预测 + pixel 细节 + 动作结果
action latent = 主体动作 + 可控状态转移 + 接触/执行意图
```

所以更合理的目标不是：

```text
z_video ≈ z_action
```

而是：

```text
z_action explains controllable part of video dynamics
```

即 action latent 是 video latent 中 controllable / actor-centric dynamics 的原因或解释变量，而不是完整 video latent 的副本。

## 5. 推荐方法：Internal World2Act for WAM

可以命名为：

```text
Asymmetric Controllable-Subspace Post-Training for WAM
```

整体分为两个阶段。

### Stage 1: Adapter / Probe 训练

冻结 LingBot-VA-SFT 主体，只训练两个 adapter 和少量辅助 head。

从每条 transition 中取：

```text
(o_t, instruction, a_t:t+k, o_t+k)
```

前向冻结的 LingBot-VA，抽取：

```text
h_v = video/world branch hidden
h_a = action branch hidden
```

更推荐使用 video delta：

```text
Δh_v = h_v(t+k) - h_v(t)
```

然后：

```text
c_v = A_v(Δh_v)   # video controllable dynamics representation
c_a = A_a(h_a)    # action-causal representation
```

Stage 1 的目标是定义一个合理的 latent supervision space，而不是直接训练最终 policy。

### Stage 2: WAM Action Path 后训练

冻结 Stage 1 中训练好的 adapters，冻结或半冻结 video branch，只更新：

```text
action head / action expert / action-video cross-attention / LoRA
```

目标是让新的 action latent 能解释 video branch 中的 controllable dynamics：

```text
A_a(h_a_new) ≈ stopgrad(A_v(Δh_v))
```

同时保留原 action loss：

```text
L_post = L_action + λ L_latent_consistency + β L_counterfactual + γ L_cycle
```

推理时可以不保留 adapter，因此不改变原模型推理架构。

## 6. Stage 1 的核心 loss

Stage 1 不建议一开始就做复杂大而全的目标。最小推荐：

```text
L_stage1 =
  L_recon_action
+ λ1 L_inverse
+ λ2 L_counterfactual
```

`L_relational` 可作为可选项或 ablation。

### 6.1 L_recon_action

目的：保证 action-side latent 保留真实动作信息，防止 action adapter 学到空表征或 shortcut。

```text
a_hat = D_a(c_a)
L_recon_action = ||a_hat - a_gt||_1 / Huber
```

如果 action 是连续控制，用 L1/Huber/MSE；如果 gripper 是离散开合，用 CE/BCE。

### 6.2 L_inverse

目的：保证 video latent 中的 controllable dynamics 确实包含 action 信息，而不是只有 pixel/context。

```text
a_from_video = D_inv(c_v)
L_inverse = ||a_from_video - a_gt||_1
```

含义不是要求 `c_v = c_a`，而是要求：

```text
c_v sufficient for action decoding
```

### 6.3 L_relational

目的：不强行点对点对齐两个 latent，而是让二者的相似性结构一致。

计算 batch 内相似度矩阵：

```text
S_v[i,j] = sim(c_v_i, c_v_j)
S_a[i,j] = sim(c_a_i, c_a_j)
```

然后：

```text
L_relational = || normalize(S_v) - normalize(S_a) ||^2
```

含义是：动作上相似的 transition，在 video controllable dynamics 上也应该相似。

### 6.4 L_counterfactual

目的：错误 action 不能匹配正确 video transition，防止模型只学 task prior 或场景 shortcut。

构造 hard negatives：

- 时间错位 action；
- 反向动作，例如 left/right、open/close、push/pull；
- gripper timing 错误；
- 同场景错误 action；
- near-miss action；
- instruction mismatch。

打分：

```text
score(i,j) = sim(c_v_i, c_a_j)
```

要求：

```text
score(c_v_i, c_a_good) > score(c_v_i, c_a_bad)
```

可以用：

```text
L_cf = -log σ((score_pos - score_neg) / τ)
```

或者 hinge loss：

```text
L_cf = max(0, margin - score_pos + score_neg)
```

## 7. 为什么不直接使用普通对比学习

不是完全不要 contrastive，而是不建议使用 naive InfoNCE：

```text
positive: (c_v_i, c_a_i)
negative: batch 中所有其他 c_a_j
```

原因：

1. batch 里可能有大量 false negatives，例如两个样本动作语义相同但场景不同；
2. video latent 和 action latent 信息不对称，普通点对点对齐容易把 action 表征污染成场景/任务 shortcut；
3. 普通 contrastive 可能学到 task ID、物体位置、成功先验，而不是真正的 action causality。

更推荐使用：

```text
conditional / counterfactual contrastive
```

即在同一个初始状态或相近状态下构造 good action 与 bad action，让模型区分哪个 action 能解释该 video transition。

## 8. 如何避免 loss 太复杂、过拟合或难学

建议采用递进实验，而不是一开始堆所有 loss：

```text
Minimal:
L_recon_action + L_inverse

Better:
+ L_counterfactual

Optional:
+ L_relational
```

主方法建议只保留三个核心：

```text
L_recon_action + L_inverse + L_counterfactual
```

逻辑清晰：

- `L_recon_action`: action latent 有动作信息；
- `L_inverse`: video latent 有动作信息；
- `L_counterfactual`: 正确/错误动作可分。

由于 Stage 1 冻结 LingBot-VA 主体，只训练小 adapter/head，过拟合风险相对可控。关键是验证集必须包含 held-out object、held-out scene、held-out instruction，而不是只看训练 benchmark 成功率。

## 9. 表征评测指标

不应只看具体任务成功率。需要 representation diagnostics。

推荐最小指标：

### 9.1 Linear Action Probe

冻结 backbone，训练小 probe：

```text
probe(Δh_video) -> action chunk
probe(Δh_video) -> end-effector delta
probe(Δh_video) -> gripper open/close
probe(Δh_video) -> contact event
```

指标：action MSE/L1、gripper accuracy、contact F1。

线性 probe 越强，说明 action 信息在 latent 里越“拉直”。

### 9.2 Action Retrieval

给定 video transition，检索正确 action：

```text
query: Δh_video
candidate: h_action_1, h_action_2, ...
```

指标：Recall@1、Recall@5、MRR、hard-negative accuracy。

### 9.3 Counterfactual Matching / AUROC

判断 action-transition 是否匹配：

```text
(Δh_v_good, h_a_good) -> positive
(Δh_v_good, h_a_bad)  -> negative
```

指标：AUROC、positive-negative margin、action swap sensitivity。

### 9.4 Nuisance Probe

检查 action latent 是否泄露背景/相机/纹理等无关信息。

指标：

- background ID probe accuracy 越低越好；
- camera ID probe accuracy 越低越好；
- nuisance CKA/HSIC 越低越好；
- controllable dynamics probe 越高越好。

### 9.5 Relational Geometry

比较 action 空间和 video controllable-delta 空间的相似性结构：

```text
D_action[i,j] = distance(h_action_i, h_action_j)
D_video[i,j]  = distance(Δh_video_ctrl_i, Δh_video_ctrl_j)
```

指标：Spearman correlation、Kendall tau、RSA、CKA。

### 9.6 Low-Shot / OOD Transfer

冻结 WAM backbone，在 target task 上只训练小 action head，用少量 demo 测：

```text
5 / 10 / 20 demos
```

指标：

- low-shot success curve；
- area under demo-efficiency curve；
- OOD object / background / camera / instruction success；
- frozen feature transfer gap。

## 10. 推荐实验对照

至少包括：

1. LingBot-VA-SFT；
2. LingBot-VA + naive contrastive alignment；
3. LingBot-VA + delta video latent alignment；
4. LingBot-VA + asymmetric controllable-subspace alignment；
5. LingBot-VA + counterfactual negatives。

关键 ablation：

- 去掉 `L_recon_action`；
- 去掉 `L_inverse`；
- 去掉 `L_counterfactual`；
- `L_relational` vs ordinary contrastive；
- full video latent vs video delta latent；
- random negatives vs hard counterfactual negatives。

## 11. 预期论文主张

可以表述为：

```text
Video co-training in WAMs provides action-relevant latent dynamics,
but success-only SFT tends to entangle these dynamics with task-success priors.
We propose a lightweight post-training method that aligns the action-causal
latent with the controllable subspace of video dynamics through inverse,
reconstruction, and counterfactual consistency losses.
```

中文：

```text
WAM 中的视频 co-training 的主要价值不是测试时显式生成未来帧，
而是在隐空间中提供 action-relevant dynamics 表征。
但普通 success-only SFT 容易把这种表征和任务成功先验纠缠在一起。
因此，我们提出一种少数据、低侵入的后训练方法，
通过 action reconstruction、inverse consistency 和 counterfactual matching，
让 action latent 对齐并解释 video latent 中的可控状态转移子空间，
从而提升表征的可解码性、可分性和 OOD/low-shot 可迁移性。
```

## 12. 当前仍需进一步明确的问题

- LingBot-VA 中具体抽取哪一层作为 `h_v` 和 `h_a`；
- video delta 是否用 token-level delta、sequence pooled delta，还是 actor/contact token；
- hard negatives 如何系统构造；
- Stage 2 是否需要 residual policy，还是只用 LoRA 更新 action path；
- evaluation 是否有足够的 OOD / failure / counterfactual 数据支持。

## 13. 已整理过的相关论文

已有中文论文整理文档：

```text
world_model_action_representation_papers_cn.md
```

涉及：

- EVA: IDM reward / executability alignment；
- WAV: state plausibility + action reachability；
- MiraBench: action-conditioned reliability / optimism bias；
- LAPA: latent action pretraining；
- CoLA-World: latent action model 与 world model 协同演化。
