# LingBot-VA RoboTwin 阶段 0 测试代码实现指南

> 适用范围：使用 `wan_va/train.py` 与 `wan_va/train_contrastive_align.py` 得到的全量 DiT checkpoint。
>
> 本文只规定阶段 0 的诊断代码，不包含阶段 1 probe、LoRA、蒸馏或新的后训练方法。

## 1. 阶段 0 要回答什么

阶段 0 不是再做一轮 benchmark，也不是用离线指标替代 rollout success。它只回答以下定位问题：

1. clean full fine-tuning 是否显著改变了 DiT 参数与中间表示；
2. contrastive checkpoint 是否真的不同于普通 clean checkpoint；
3. post-trained 模型是否更依赖亮度、颜色、背景等不应改变动作的视觉因素；
4. 这种敏感性来自 world/video branch，还是主要来自 action readout；
5. clean 与 C2R 的表示距离是否在 post-training 后被拉大。

当前实际训练入口都会更新 DiT：

- `wan_va/train.py` 将 `self.transformer.requires_grad_(True)`；
- `wan_va/train_contrastive_align.py` 继承同一个 `Trainer`，同样更新全部 transformer 参数，并额外训练 `align_head`；
- 两者都把完整 DiT 保存到 `checkpoint_step_x/transformer/`。

因此，阶段 0 的主要比较对象是：

```text
theta_base         = lingbot-va-base/transformer
theta_clean        = train.py 产出的 checkpoint_step_x/transformer
theta_contrastive  = train_contrastive_align.py 产出的 checkpoint_step_x/transformer
theta_aug          = 可选的 augment checkpoint，作为正对照
```

`theta_contrastive` 的 `align_head/` 只在训练时提供辅助监督，不是最终 policy 的一部分。DiT 漂移、action sensitivity 与 rollout 诊断均加载 `transformer/`；`align_head/` 只用于核验 contrastive 训练是否真正生效。

## 2. 总体实现原则

### 2.1 必须同时保留两条测量路径

阶段 0 需要区分：

| 路径 | 用途 | 是否作为最终行为结论 |
|---|---|---:|
| Deterministic train-forward | 逐层 feature、CKA、有限差分和低成本 action velocity 代理 | 否，主要用于定位 |
| Deterministic policy inference | 完整 video/action denoising、最终 action chunk 与 future video | 是 |

原因是 `forward_train` 与实际推理并不完全相同：

- `forward_train` 同时拼接 noisy video、condition video、noisy action、condition action；
- 训练数据里的 `condition action` 来自 ground-truth action；
- 实际 server 先运行 video denoising、写入 KV cache，再运行 action denoising；
- 因此只在 `forward_train` 上测 action sensitivity，可能低估真实 policy 对视觉扰动的依赖。

正确做法是：先用 train-forward 在较大样本上快速定位，再用实际 inference path 在较小样本上确认。

### 2.2 所有 checkpoint 必须共享完全相同的随机量

每个样本的以下内容必须由 `sample_id` 确定，并在 checkpoint 间复用：

- video noise；
- action noise；
- video timestep id；
- action timestep id；
- `chunk_size`；
- `window_size`；
- CFG negative prompt；
- full denoising 的初始噪声；
- augmentation 参数。

禁止直接调用训练代码中的随机 `_prepare_input_dict()`，因为它会随机：

- 每帧 timestep；
- Gaussian noise；
- condition video 是否加噪；
- `chunk_size in [1, 4]`；
- `window_size in [4, 64]`。

诊断代码应实现单独的 `DeterministicInputBuilder`，但内部仍复用训练的 `FlowMatchScheduler`、`get_mesh_id()` 和张量布局。

### 2.3 不要把 C2R 数据用于调参

C2R observation 只能用于最终诊断和报告，不得用于：

- 选择 checkpoint step；
- 选择 contrastive lambda；
- 选择 hook layer；
- 选择扰动强度；
- early stopping；
- 选择模型融合系数。

所有上述选择应在 clean holdout 或 clean-derived corruption validation 上完成。否则阶段 0 会把 C2R benchmark 变成开发集。

## 3. 推荐代码目录

建议新增独立目录，不要把诊断逻辑塞入训练类：

```text
evaluation/robotwin/stage0/
├── configs/
│   └── robotwin_stage0.yaml
├── common/
│   ├── checkpoint.py
│   ├── deterministic.py
│   ├── feature_hooks.py
│   ├── observation_encoder.py
│   ├── perturbations.py
│   ├── policy_harness.py
│   ├── metrics.py
│   └── io.py
├── build_clean_probe_manifest.py
├── build_paired_latent_bank.py
├── collect_c2r_observation_bank.py
├── audit_checkpoints.py
├── extract_train_features.py
├── measure_action_sensitivity.py
├── measure_video_dynamics.py
├── measure_directional_jacobian.py
├── measure_domain_gap.py
└── aggregate_stage0_report.py
```

第一版不必一次写完全部脚本。推荐实现顺序见第 14 节。

## 4. 配置文件规范

`robotwin_stage0.yaml` 至少应包含：

```yaml
checkpoints:
  base: /path/to/lingbot-va-base/transformer
  clean: /path/to/clean/checkpoint_step_5000/transformer
  contrastive: /path/to/contrastive/checkpoint_step_5000/transformer
  # aug: /path/to/augment/checkpoint_step_5000/transformer

model_root: /path/to/lingbot-va-base
clean_dataset_path: /path/to/lerobot_robotwin_eef_clean_50
clean_manifest: /path/to/stage0_clean_probe.jsonl
c2r_bank: /path/to/stage0_c2r_bank
output_dir: output/robotwin_stage0

runtime:
  dtype: bfloat16
  metric_dtype: float32
  attn_mode: flex
  batch_size: 1
  seed: 20260629

train_forward:
  chunk_size: 2
  window_size: 64
  condition_video_noise: false
  timestep_quantiles: [0.25, 0.50, 0.75]
  primary_timestep_quantile: 0.50
  hook_layers: auto

policy_inference:
  video_steps: 25
  action_steps: 50
  guidance_scale: 5.0
  action_guidance_scale: 1.0

perturbations:
  brightness: [0.80, 1.20]
  gamma: [0.80, 1.20]
  color_temperature: [-0.10, 0.10]
  gaussian_noise_std: [0.01, 0.03]
  gaussian_blur_sigma: [0.5, 1.0]

statistics:
  bootstrap_repeats: 2000
  confidence_level: 0.95
  cluster_unit: episode
```

注意：`timestep_quantiles` 表示 scheduler 索引分位点，不要把 `0.5` 直接当作模型 timestep。video scheduler 使用 `snr_shift=5.0`，action scheduler 使用 `action_snr_shift=1.0`，应分别构造 scheduler，再由索引取得实际 `timesteps` 与 `sigmas`。

## 5. 数据准备

### 5.1 Clean probe manifest

不要依赖 DataLoader shuffle 后的位置。每个 probe window 必须有稳定 ID：

```json
{
  "sample_id": "clean/pick_apple_messy/episode_000012/120_184",
  "domain": "clean",
  "task": "pick_apple_messy",
  "repo_id": "/path/to/task_dataset",
  "episode_index": 12,
  "start_frame": 120,
  "end_frame": 184,
  "action_config_index": 0,
  "split": "clean_holdout"
}
```

数据优先级：

1. 最优：重新生成未用于训练的 clean expert episodes；
2. 次优：训练前预留的 episode-level holdout；
3. 仅用于已有 checkpoint 的临时诊断：从训练集固定抽样，但必须在报告中标记为 `clean_train_probe`，不能称为 validation。

严禁按 frame 随机切分，因为同一 episode 的相邻窗口高度相关。应按 episode 切分和采样。

建议规模：

- smoke test：10 tasks，每任务 5 episodes，每 episode 2 windows；
- 正式离线诊断：50 tasks，每任务至少 20 windows；
- 同一 task 不应由少数长 episode 垄断样本数。

### 5.2 禁止文本 CFG 随机丢弃

`LatentLeRobotDataset._cat_video_latents()` 会按 `cfg_prob` 随机把任务文本替换成 `empty_emb`。阶段 0 必须：

- 令诊断 config 的 `cfg_prob=0.0`；
- 或在 diagnostic dataset wrapper 中直接读取原始 `text_emb`；
- 将 prompt hash 写入样本 metadata；
- 同一个样本对所有 checkpoint 使用同一个 text embedding。

### 5.3 Paired RGB/latent bank

现有训练数据接口只返回预编码 latent。brightness、gamma、color temperature、blur 等扰动必须发生在 RGB 空间，之后重新走同一 VAE 编码路径。

应从 `wan_va/wan_va_server.py::_encode_obs()` 抽取一个无 server 副作用的 `RobotwinObservationEncoder`，保持以下流程完全一致：

1. high camera resize 到 `256 x 320`；
2. left/right wrist resize 到 `128 x 160`；
3. RGB 从 `[0, 255]` 映射到 `[-1, 1]`；
4. high camera 使用完整 VAE streaming encoder；
5. 两个 wrist camera 使用 half-resolution streaming encoder；
6. 左右 wrist latent 在 width 维拼接；
7. wrist 与 high latent 在 height 维拼成 T-shape；
8. 使用 VAE `latents_mean`、`latents_std` 做与 server 一致的 normalize；
9. 使用 posterior mean `mu`，不要在 VAE posterior 中额外采样。

每个 clean clip 生成：

```text
clean RGB -> z_clean
T_k(clean RGB; transform_seed) -> z_aug_k
```

同一种扰动在一个 clip 的所有帧上参数保持一致。默认也对三路相机使用同一参数，模拟全局照明变化；另设 `per_camera_sensor_shift` 组，才允许相机间参数不同。

只在 latent 上加 Gaussian noise 可以作为代码 smoke test，但不能作为 photometric robustness 的主要结果。

### 5.4 C2R observation bank

C2R bank 不应来自某个待测 policy 的失败轨迹，否则不同模型看到的状态分布不同。推荐复用：

- `rt_c2r_manifests/*.jsonl` 的 validated seeds；
- `generate_rt_c2r_validated_manifests.py` 中的 `build_task_args()` 与 `setup_demo()`；
- RoboTwin scripted expert 的 `play_once()`。

采集方式：

1. 使用同一批 expert-valid seeds；
2. 在 `setup_demo()` 后保存初始 observation；
3. 在 scripted expert 执行期间按固定 progress bin 保存 observation；
4. 推荐 progress bins 为 `0.0/0.25/0.5/0.75/1.0`；
5. 每个 observation 保存三路 RGB、EEF state、task、split、seed、progress bin；
6. 再通过统一 `RobotwinObservationEncoder` 编码。

若 RoboTwin expert API 不方便暴露 progress，可先只采集 `initial` 与 `final`。初始帧状态匹配最可靠，应该单独报告；不要把初始帧和完成态混在一个 domain distance 中。

C2R bank 建议的最小 schema：

```json
{
  "sample_id": "c2r/background/pick_apple_messy/seed_123/progress_025",
  "domain": "c2r",
  "split": "background",
  "task": "pick_apple_messy",
  "seed": 123,
  "progress_bin": 0.25,
  "expert_validated": true,
  "prompt": "...",
  "rgb_paths": {
    "cam_high": "...",
    "cam_left_wrist": "...",
    "cam_right_wrist": "..."
  },
  "latent_path": "..."
}
```

## 6. Checkpoint audit：所有指标之前的硬门槛

脚本：`audit_checkpoints.py`

### 6.1 加载规则

统一使用：

```python
load_transformer(
    transformer_path,
    torch_dtype=torch.float32,
    torch_device="cpu",
    attn_mode="flex",
)
```

之后移动到单 GPU、切换 `eval()` 并关闭梯度。阶段 0 不需要 FSDP，也不要调用训练 `Trainer`，否则会连带创建 optimizer、DataLoader 和随机输入。

### 6.2 参数指纹

对每个 post-trained checkpoint 与 base 逐参数流式计算：

```text
relative_l2(name) = ||theta_ft[name] - theta_0[name]||_2
                    / (||theta_0[name]||_2 + eps)

update_cos(name)  = cosine(theta_ft[name] - theta_0[name], theta_0[name])
```

至少按以下模块聚合：

- `patch_embedding_mlp`；
- `action_embedder`；
- `condition_embedder`；
- `condition_embedder_action`；
- 每个 `blocks.i`；
- `norm_out`；
- `proj_out`；
- `action_proj_out`。

同时记录：

- 权重文件 SHA256；
- transformer config SHA256；
- checkpoint step；
- parameter name/shape mismatch；
- 非有限值数量。

`align_head/` 单独记录文件 hash、参数范数与 trainer state 中的 `lambda_align`，但不得并入 DiT 参数距离。

### 6.3 通过条件

- base 与自己的 relative L2 必须为 0；
- clean/contrastive 的 transformer 应有非零更新；
- 所有 checkpoint 参数名和 shape 一致；
- 不允许 NaN/Inf；
- 若 contrastive 与 clean 的 transformer 几乎一致，应进一步检查训练日志中的 `align_loss`、有效 pair 数和 align head gradient，不能直接宣称 contrastive 无效。

这一步的输出为 `metrics/checkpoint_audit.json` 和 `plots/parameter_drift_by_layer.png`。

## 7. Deterministic train-forward

脚本公共组件：`common/deterministic.py`

### 7.1 输入构造

输入张量沿用训练数据形状：

```text
latents      [B, 48, F, H, W]
actions      [B, 30, F, 16, 1]
actions_mask [B, 30, F, 16, 1]
text_emb     [B, L_text, D_text]
```

使用与训练相同的两个 scheduler：

```python
video_scheduler = FlowMatchScheduler(
    shift=5.0, sigma_min=0.0, extra_one_step=True
)
video_scheduler.set_timesteps(1000, training=True)

action_scheduler = FlowMatchScheduler(
    shift=1.0, sigma_min=0.0, extra_one_step=True
)
action_scheduler.set_timesteps(1000, training=True)
```

确定性噪声 seed 不应依赖 Python 内置 `hash()`，因为它跨进程不稳定。使用：

```text
seed = first_64_bits(SHA256(global_seed + sample_id + stream_name))
```

`stream_name` 至少区分 `video_noise`、`action_noise`、`augmentation` 和 `policy_noise`。

主 CKA schedule：

- video/action 均使用 scheduler 50% 索引对应的 timestep；
- 所有帧使用同一 timestep；
- `cond_timesteps=0`；
- `chunk_size=2`；
- `window_size=64`；
- `actions_mask` 与训练一致；
- `model.eval()`；
- batch size 第一版固定为 1。

action sensitivity 额外在 25%、50%、75% 三个 timestep quantile 上重复。结果中必须记录 quantile、实际 timestep 和 sigma。

### 7.2 为什么不直接调用 `_add_noise()`

`Trainer._add_noise()` 内部自行采样 timestep 和 noise，并且 `_prepare_input_dict()` 会以 0.5 概率污染 condition video。即使每次手动 `torch.manual_seed()`，多 checkpoint、分布式进程或新增一次随机调用后也容易错位。

诊断 builder 应显式接收：

```python
build_train_input(
    batch,
    video_noise,
    action_noise,
    video_timestep_ids,
    action_timestep_ids,
    chunk_size,
    window_size,
    condition_video_noise=None,
) -> dict
```

## 8. 分层 feature hook

脚本：`extract_train_features.py`

### 8.1 现有 `return_hidden=True` 不够

`WanTransformer3DModel.forward_train()` 的 token 顺序是：

```text
[noisy video] [condition video] [noisy action] [condition action] [padding]
```

但 `return_hidden=True` 只返回最终 `norm_out` 后的：

- noisy video hidden；
- noisy action hidden。

它不返回 condition video，也不返回中间层。因此逐层 CKA 必须注册：

```python
handle = transformer.blocks[layer_idx].register_forward_hook(hook)
```

`WanTransformerBlock.forward()` 直接返回一个 tensor，hook 应读取 `output`，而不是假定它是 tuple。

### 8.2 自动选择层

不要硬编码 30 层。令：

```text
L = len(transformer.blocks)
layers = sorted(unique([0, round((L-1)/3), round(2*(L-1)/3), L-1]))
```

若当前模型确实为 30 层，对应近似 `0/10/19/29`。最终报告必须写实际 layer index。

### 8.3 Token slice 计算

设：

```text
Lv = (Fv / patch_f) * (Hv / patch_h) * (Wv / patch_w)
La = Fa * Ha * Wa
```

当前 RoboTwin `patch_size=(1,2,2)`，action patch size 视为 `(1,1,1)`。

由于每组 tensor 都先执行 `.flatten(0, 1)[None]`，batch size 为 `B` 时的 offset 为：

```text
noisy_video:      [0,          B*Lv)
condition_video:  [B*Lv,       2*B*Lv)
noisy_action:     [2*B*Lv,     2*B*Lv+B*La)
condition_action: [2*B*Lv+B*La, 2*B*Lv+2*B*La)
padding:          [remaining]
```

每个 slice 再 reshape 为 `[B, L_group, C]`。绝对不能把 padding token 纳入 feature，也不能把四类 token 混在一个 CKA 矩阵中。

### 8.4 Pooling 与保存

每层至少保存两种 feature：

```text
condition_video_window: [B, C]
condition_video_frame:  [B, F_patch, C]
```

计算方式：

1. condition video 恢复为 `[B, F_patch, H_patch, W_patch, C]`；
2. frame feature 对空间 token 求均值；
3. window feature再对时间求均值；
4. 转为 float32 后存 CPU；
5. 同时保存 noisy video 与 noisy action pooled feature，作为辅助定位；
6. CKA 的主结果只使用 condition video。

保存前检查每个 checkpoint 的 `sample_id` 顺序完全相同。feature 文件中必须同时写入 `sample_ids`，不能依赖文件行号隐式配对。

## 9. 指标一：CKA 与 cosine feature drift

脚本：`extract_train_features.py` + `common/metrics.py`

### 9.1 输入

对每个选定层构造：

```text
X_base [N, C]
X_ft   [N, C]
```

其中 N 为相同 `sample_id` 的 window feature。主结果使用 window-level feature；frame-level feature作为补充。不要把大量相邻 spatial token 当作独立样本，否则置信区间会虚假变窄。

### 9.2 计算

Linear CKA：

```text
CKA(X, Y) = ||X_c^T Y_c||_F^2
            / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F + eps)
```

其中 `X_c`、`Y_c` 沿样本维中心化。矩阵乘法和累加至少使用 float32；如果显存允许，最终 Gram/HSIC 累加使用 float64。

Cosine drift：

```text
Drift_cos = 1 - mean_i cosine(x_base_i, x_ft_i)
```

输出：

- 每层 CKA；
- 每层 mean/median cosine drift；
- 按 task 分组的 CKA 或 cosine drift；
- `clean-base`、`contrastive-base`、可选 `aug-base`；
- 95% cluster bootstrap CI。

### 9.3 Sanity checks

- base 对 base：`CKA > 0.9999`，cosine drift 接近 0；
- 同一 feature 文件重复加载结果完全一致；
- 随机打乱 `X_ft` 的 sample order 后，CKA 应明显下降；
- 删除某个 sample 后必须通过 ID join 对齐，不能静默错位；
- 早中后层都报告，不允许只挑最符合假设的一层。

CKA 只表示表示几何变化，不是能力分数。`CKA` 下降本身不能证明泛化变差，必须与后续 sensitivity、domain gap 和 rollout 联合解释。

## 10. 指标二：Action sensitivity

脚本：`measure_action_sensitivity.py`

### 10.1 扰动集合

主扰动必须满足“不改变该状态下正确动作”的近似假设：

- clip-consistent brightness；
- gamma；
- color temperature；
- 轻度 Gaussian noise；
- 轻度 blur；
- 只替换非 robot/非 target 区域的 background，若有可靠 mask。

以下变化不能当作 nuisance invariance：

- camera extrinsic；
- table/object height；
- object position；
- spatial crop/translation；
- robot/target ROI 遮挡。

这些变化可能要求动作改变，应归入 geometry/equivariance 组单独解释。

### 10.2 A 级：train-forward velocity sensitivity

对 clean 与 paired augmented latent 使用完全相同：

- noisy action `a_tau`；
- action timestep；
- noisy video 与 video timestep；
- condition action；
- text embedding；
- chunk/window 参数。

取得 action velocity output `v_a`，按 `actions_mask` 过滤后计算：

```text
S_velocity(T) = mean(
    ||v_a(T(x)) - v_a(x)||_2
    / (||v_a(x)||_2 + eps)
)
```

同时报告：

- left/right EEF position channels；
- rotation channels；
- gripper channels；
- 每个 action frame；
- 25%/50%/75% timestep quantile；
- task-level 分布。

这是低成本定位指标，不作为最终 action robustness 结论。

### 10.3 B 级：真实 policy-path sensitivity

必须实现一个确定性的 `policy_harness.py`，复用 `wan_va_server.py` 的真实顺序：

```text
encode observation
-> 25-step video denoising
-> cache final video tokens
-> 50-step action denoising
-> postprocess action
```

clean 与 augmented 分支必须：

- 使用相同初始 video noise；
- 使用相同初始 action noise；
- 使用相同 prompt 与 CFG negative prompt；
- 每个分支开始前清空所有 attention KV/pred cache；
- 不共享上一样本的 streaming VAE cache；
- 使用相同 inference step 数与 guidance scale；
- 保存每一步 action velocity，便于定位敏感性从何时出现。

最终 action chunk 计算：

```text
S_action_L1 = mean(|a_hat_aug - a_hat_clean| / action_scale)
S_action_L2 = ||a_hat_aug - a_hat_clean||_2 / (||a_hat_clean||_2 + eps)
```

还应报告：

- gripper open/close flip rate；
- EEF position endpoint difference；
- rotation geodesic difference；
- temporal jerk difference；
- action chunk 前 25%、后 75% 分段结果。

第一版 policy-path confirmation 可只跑 10 tasks x 5 observations x 每种扰动 2 个强度，避免完整 25+50 步推理成本失控。

### 10.4 Identity test

增加 `T_identity`：重新走一遍相同编码与推理，但不改变像素。它的差异应接近数值噪声。如果 identity sensitivity 已经很大，应先修复 cache、seed 或 VAE streaming 状态，不能解释其他结果。

## 11. 指标三：Future-video dynamics stability

脚本：`measure_video_dynamics.py`

这项指标使用 policy inference harness 中已经生成的 future latent，不建议额外维护另一套生成代码。

### 11.1 不使用 raw future latent MSE 作为主指标

亮度和色温会改变 VAE latent 的 style component，因此：

```text
MSE(z_future_clean, z_future_aug)
```

会把合理的外观差异误判为动力学错误。

主比较对象是 temporal residual：

```text
Delta_z_clean = z_future_clean - z_current_clean
Delta_z_aug   = z_future_aug   - z_current_aug

S_dyn_latent = 1 - cosine(
    whiten(Delta_z_clean),
    whiten(Delta_z_aug)
)
```

whitening 统计只能从 clean probe bank 估计，并冻结后用于所有 checkpoint。

### 11.2 三层输出

按成本从低到高实现：

1. `latent_residual_cosine`：必须实现；
2. `semantic_motion_distance`：解码后用固定外部视觉 encoder 比较逐帧 feature displacement；
3. `motion_structure_distance`：比较 optical flow、robot/object keypoint 或 segmentation centroid trajectory。

若暂时没有稳定的 ROI/keypoint 模型，第一版只做第 1 项，并在报告中明确它仍受 VAE 表示影响。不要用不可靠的伪 mask 制造更复杂但更难解释的指标。

### 11.3 联合 action 解释

保存同一 paired run 的 `S_dyn` 与 `S_action`，按 sample 关联：

- `S_dyn` 低、`S_action` 高：world/video prior 较稳定，问题更可能在 action 路径；
- 两者都高：视觉扰动已进入 world dynamics；
- 两者都低但 rollout 失败：可能不是 photometric robustness 问题；
- `S_action` 过低且 geometry 扰动也无响应：可能形成过度不变性。

## 12. 指标四：Directional finite difference

脚本：`measure_directional_jacobian.py`

完整像素 Jacobian 对当前模型过于昂贵。实现有限差分：

```text
J_FD(x; delta) = ||f(x + epsilon * delta) - f(x)||_2 / epsilon
```

其中 `f` 第一版使用 train-forward action velocity；对少量样本再使用 policy-path 最终 action chunk。

### 12.1 方向分组

```text
delta_photo: brightness / gamma / color direction
delta_bg:    只作用于背景区域
delta_roi:   只作用于 robot 或 target ROI
delta_geom:  小幅 spatial shift 或 camera-like warp
```

每个方向先归一化为单位 L2 norm，再乘 `epsilon`。每个方向至少测试正负两侧，使用中心差分更稳定：

```text
J_centered = ||f(x + eps*delta) - f(x - eps*delta)||_2 / (2*eps)
```

至少检查两个 epsilon，例如 `0.5 eps` 与 `eps`。若估计值相差很大，说明不在局部线性区或数值精度不足。

### 12.2 期望模式

不能简单追求所有 Jacobian 越小越好。更合理的诊断量是：

```text
R_nuisance = mean(J_photo, J_bg)
R_signal   = mean(J_roi, J_geom)
selectivity = R_signal / (R_nuisance + eps)
```

理想方向是 nuisance sensitivity 下降，同时保留 ROI/geometry 响应。只报告整体 Gaussian direction norm 会混合两类方向，不足以支持结论。

## 13. 指标五：Clean-C2R domain gap

脚本：`measure_domain_gap.py`

### 13.1 Feature 选择与匹配

主 feature 使用与 CKA 相同的 condition-video pooled hidden。比较每个 checkpoint 的同一组层。

clean 与 C2R 应按以下键分层或匹配：

- task；
- progress bin；
- 尽量相同的 seed/object initialization；
- C2R split；
- observation type：initial/intermediate/final。

至少把 `background/light/clutter/height` 分开报告。geometry split 与 photometric split 不应混为一个平均数。

### 13.2 kNN distance

先对 feature 做 L2 normalization，按 `task + progress_bin` 建立 clean bank：

```text
D_knn = mean_{x in C2R} min_{y in CleanBank(task, stage)} ||h(x)-h(y)||_2
```

同时记录 k=1 与 k=5 的平均距离，防止单个异常近邻主导结果。

### 13.3 MMD

使用 RBF kernel：

```text
MMD^2(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
```

kernel bandwidth 使用 clean feature pairwise distance 的固定 median heuristic。不要为每个 checkpoint 单独寻找最优 bandwidth，否则数值不可比。

### 13.4 Domain linear separability

冻结 feature，训练一个只用于分析的 logistic regression 预测 clean/C2R：

- class balance 为 1:1；
- 固定 `C=1.0`；
- train/test 按 episode 或 seed group split；
- 同一 trajectory 的 frame 不得跨 train/test；
- 报告 balanced accuracy 与 AUROC；
- classifier 参数绝不能进入 policy。

### 13.5 相对 base 的变化

所有 domain metric 最终报告：

```text
Delta_D(checkpoint) = D(checkpoint) - D(base)
```

解释重点不是某个绝对阈值，而是 clean 与 contrastive 后训练相对 base 是否把 gap 拉大，以及这种变化是否与 C2R rollout success/action sensitivity 一致。

## 14. 推荐实现顺序

### M0：确定性与 checkpoint 审计

必须先完成：

- config loader；
- checkpoint loader；
- parameter fingerprint；
- stable sample ID；
- deterministic noise/timestep builder；
- identity repeat test。

M0 未通过时，不应继续计算任何正式指标。

### M1：逐层 feature 与 CKA

实现：

- block hooks；
- 四组 token slicing；
- condition-video pooling；
- CKA/cosine drift；
- task/episode cluster bootstrap。

这是最小可用阶段 0 的核心。

### M2：paired RGB bank 与 action sensitivity

实现：

- 统一 observation encoder；
- clip-consistent photometric transforms；
- train-forward velocity sensitivity；
- 真实 policy-path 小规模确认；
- gripper/position/rotation 分项。

### M3：C2R observation bank 与 domain gap

实现：

- scripted-expert observation collector；
- initial/progress 分层；
- kNN、MMD、linear separability；
- C2R 数据隔离检查。

### M4：future dynamics 与 directional finite difference

复用 M2 的 inference harness，不再创建第三套模型执行代码。

## 15. 输出目录与数据格式

每次运行生成不可覆盖的 `run_id`：

```text
output/robotwin_stage0/<run_id>/
├── config_resolved.yaml
├── environment.json
├── manifests/
├── checkpoint_audit/
├── features/
│   ├── base/
│   ├── clean/
│   └── contrastive/
├── paired_predictions/
├── metrics/
│   ├── feature_drift.csv
│   ├── action_sensitivity.csv
│   ├── video_dynamics.csv
│   ├── directional_fd.csv
│   └── domain_gap.csv
├── plots/
└── report.md
```

`environment.json` 至少记录：

- git commit 与 dirty status；
- PyTorch/CUDA/diffusers 版本；
- GPU 型号；
- dtype 与 attention mode；
- checkpoint hash；
- model config hash；
- clean/C2R manifest hash。

每条 metric row 至少带：

```text
run_id, checkpoint, sample_id, task, episode_or_seed,
domain, c2r_split, progress_bin, layer, token_group,
perturbation, severity, timestep_quantile, timestep, sigma,
noise_seed, transform_seed, metric_name, metric_value
```

聚合脚本只能从这些长表生成结果，不应依赖进程内临时均值。

## 16. 统计与可视化

### 16.1 统计单位

frame 不是独立样本。默认置信区间按 episode cluster bootstrap：

1. 在 task 内重采样 episode；
2. 再对 task 做等权平均；
3. 重复 2000 次；
4. 报告 95% CI。

checkpoint 比较使用 paired difference，因为它们处理相同 sample、noise 和 transform。

### 16.2 必须生成的图

- parameter relative L2 by layer/module；
- CKA vs layer；
- cosine drift vs layer；
- action sensitivity 按 perturbation 和 timestep 分组；
- clean/C2R kNN 或 MMD gap vs layer；
- `S_dyn` 对 `S_action` 的 sample scatter；
- 每个 C2R split 的相对 base 变化；
- task-level distribution，而不只给一个总体均值。

所有图都同时画 base、clean、contrastive，可选 aug。不能只展示 clean 与 contrastive 的相对差异而隐藏 base。

## 17. 验收测试

### 17.1 数值与确定性

- 相同 checkpoint、相同输入运行两次，sample ID 与随机 metadata 完全相同；
- base-vs-base CKA 大于 0.9999；
- identity transform 的 action/video difference 接近数值误差；
- 同一个 noise tensor 的 SHA256 在 checkpoint 间相同；
- 所有 metric 无 NaN/Inf；
- padding token 不进入 pooling；
- action metric 严格应用 `actions_mask`。

BF16/flex attention 下逐元素输出不一定 bitwise 相等。若 identity test 失败，应先在单 GPU、batch size 1、同一进程内复查；可用 CKA、相对误差和最终 action 误差联合设容差，不要只用 `torch.equal()`。

### 17.2 数据泄漏

- manifest 明确标记 `clean_train_probe` 或 `clean_holdout`；
- 同一 episode 不跨数据 split；
- C2R seed 不参与任何 hyperparameter 选择；
- domain classifier 的同一 trajectory 不跨 train/test；
- transform severity 在读取 C2R 结果前冻结。

### 17.3 语义正确性

- RGB 三相机布局与 server 编码结果在 identity case 上一致；
- 从原始 RGB 编码得到的 clean latent，与已有预编码 latent 在容差内相符；
- perturbation 的首帧/末帧参数一致；
- geometry perturbation 不被误标为 nuisance；
- inference paired branches 每次都清空 KV cache 和 VAE streaming cache。

## 18. 最终报告模板

`aggregate_stage0_report.py` 生成的 `report.md` 应按以下顺序写：

```markdown
# Stage 0 Diagnostic Report

## Checkpoints and data
## Determinism and sanity checks
## Parameter update audit
## Layer-wise feature drift
## Action sensitivity
## Future-video dynamics stability
## Directional finite difference
## Clean-C2R domain gap
## Joint diagnosis
## Limitations
```

Joint diagnosis 使用固定规则，不根据结果临时改口径：

| 结果组合 | 优先解释 |
|---|---|
| DiT CKA 明显降低，Clean-C2R gap 增大，C2R success 下降 | full FT 覆盖了部分 pretrained world representation |
| CKA 变化小，但 action sensitivity/Jacobian 增大 | action output path 放大了已有 domain feature |
| video dynamics 稳定，action sensitivity 高 | world prior 尚在，action grounding/readout 有问题 |
| clean 与 contrastive 几乎所有指标一致，且 align loss/有效 pair 异常 | contrastive recipe 可能没有真正产生训练信号 |
| contrastive 降低 photometric sensitivity，但 geometry split 不改善 | 学到了外观不变性，没有获得 3D equivariance |
| nuisance 与 geometry sensitivity 都很低 | 可能过度不变，不能视为泛化改善 |
| 离线指标改善但 rollout 不改善 | 指标未覆盖闭环误差累积，不能据此宣称方法有效 |

## 19. 阶段 0 的完成标准

满足以下条件后，才算阶段 0 完成：

1. 已确认 base、clean、contrastive 的 transformer 权重关系；
2. 已在相同 clean samples 上得到可复现的逐层 CKA/cosine drift；
3. 已在 RGB-space paired perturbations 上测量 action sensitivity；
4. 已用真实 policy inference path 对小规模样本复核；
5. 已区分 future-video instability 与 action-only instability；
6. 已在冻结的 C2R observation bank 上测量 domain gap；
7. 所有结果都包含 base、置信区间、task/split 分解和 sanity checks；
8. C2R 数据没有参与模型或超参数选择；
9. 最终结论与 C2R rollout success 联合报告，而不是由单个离线指标推出。

在阶段 0 结束前，不建议直接进入新的 loss 设计。这里最重要的产物不是“哪个离线分数最好”，而是把失败定位到以下两类中的哪一类：

```text
A. full fine-tuning 改坏了 DiT/world representation；
B. DiT prior 仍在，但 action path 学会了 clean-domain shortcut。
```

两类问题需要完全不同的阶段 1 方案，因此必须先由上述测试把它们分开。
