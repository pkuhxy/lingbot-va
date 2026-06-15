# 架构

## `wan_va/` 代码地图

### 入口文件

- `wan_va/train.py`
  - 从 `VA_CONFIGS` 读取配置并创建 `Trainer(config)`。
  - 只加载 transformer：来源是 `resume_from/transformer` 或 `wan22_pretrained_model_name_or_path/transformer`。
  - 训练时强制 `attn_mode="flex"`。
  - 对 transformer 应用 activation checkpointing 和 FSDP 分片。
  - 数据来自 `MultiLatentLeRobotDataset`，多卡时使用 `DistributedSampler`。
  - 为视频 latent 和 action 分别创建 `FlowMatchScheduler`。
  - checkpoint 保存到 `checkpoints/checkpoint_step_<step>/transformer/`，格式与 Diffusers transformer 目录兼容。

- `wan_va/train_latent_posttrain.py`
  - Stage-1 asymmetric controllable-subspace post-training 入口。
  - 从 `VA_CONFIGS` 读取 `robotwin_latent_posttrain` 或 `libero_latent_posttrain` 配置。
  - 加载并冻结 pretrained transformer，只训练 `WAMControllableSubspaceProbe`。
  - 使用 `MultiLatentLeRobotDataset` 产出 `latents`、`actions`、`actions_mask`、`text_emb`。
  - 支持从 transformer hidden 或 input embedding 抽取 video/action tokens，默认 `latent_hidden_source="embed"`。
  - checkpoint 保存到 `<save_root>/latent_posttrain_stage1/checkpoint_step_<step>/stage1_probe.pt`，只包含 probe、optimizer、step 和 config。

- `wan_va/wan_va_server.py`
  - 创建 `VA_Server(config)`。
  - 从配置的 pretrained model path 加载 VAE、tokenizer、text encoder 和 transformer。
  - 推理时强制 transformer `attn_mode="torch"`。
  - `infer_mode == "server"` 时启动 websocket 服务；`infer_mode == "i2va"` 时从初始图片和 prompt 离线生成视频/action。
  - 维护名为 `pos` 的 transformer KV cache。
  - 请求里带 `save_visualization=True` 时，可把预测 latent 解码成视频并随 action 返回。

### 模型层

- `wan_va/modules/model.py`
  - 定义核心模型 `WanTransformer3DModel`。
  - 复用 Wan 风格 transformer block，并额外支持 action token。
  - 视频 latent 通过 `patch_embedding_mlp` 变成 token；action 通过 `action_embedder` 变成 token。
  - 视频和 action 使用两套 time/text 条件嵌入：`condition_embedder` 和 `condition_embedder_action`。
  - `forward_train()` 拼接四类 token：noisy latent、clean/condition latent、noisy action、clean/condition action。
  - `FlexAttnFunc.init_mask()` 为训练构造因果、clean/noisy、chunk、滑动窗口相关 mask。
  - 推理 `forward()` 支持 latent mode 或 action mode，并可更新/使用 attention KV cache。

- `wan_va/modules/utils.py`
  - 封装 Diffusers Wan VAE、UMT5 text encoder、T5 tokenizer 和项目 transformer 的加载。
  - `WanVAEStreamingWrapper` 保存因果 VAE encoder 的 feature cache，用于分 chunk 编码。
  - `patchify()` 在 VAE 配置存在 patch size 时调整输入。

- `wan_va/modules/latent_posttraining.py`
  - 定义 `WAMControllableSubspaceProbe`，包含 video adapter、action adapter、action decoder 和 inverse decoder。
  - `video_adapter` 把 video delta tokens 映射到 controllable dynamics subspace。
  - `action_adapter` 把 action tokens 映射到 action-causal subspace。
  - `action_decoder` 从 action-side subspace 重构逐 token action，约束 action 表征不能退化。
  - `inverse_decoder` 从 video-side pooled subspace 预测平均 action，约束 video dynamics 包含可控动作信息。
  - 提供 `counterfactual_loss()`、`relational_loss()`、`stage2_latent_consistency_loss()` 和 retrieval metrics；当前 Stage-1 trainer 使用 action recon、inverse、counterfactual、可选 relational。

### 数据集层

- `wan_va/dataset/lerobot_latent_dataset.py`
  - `MultiLatentLeRobotDataset` 扫描 `config.dataset_path` 下的 LeRobot `meta/info.json`，把多个数据集拼成一个 dataset。
  - `LatentLeRobotDataset` 继承 `LeRobotDataset`，通过 `LeRobotDatasetMetadata` 读取 episode metadata。
  - `parse_meta()` 把每个 episode 的 `action_config` 展开成可训练片段。
  - `_check_meta()` 过滤缺失 camera latent 文件的片段。
  - `_cat_video_latents()` 从 `.pth` latent 文件恢复 `(f, h, w, c)`，再按相机维度拼接。`robotwin_tshape` 有特殊拼接方式，因为腕部相机和高位相机的布局不是普通横向拼接。
  - `_action_post_process()` 将 action 对齐到 latent frame，RoboTwin 场景下可转相对 end-effector pose，随后补齐到 30 维、按分位数归一化、clip，并返回 action tensor 和 mask。
  - `__getitem__()` 输出 `latents`、`text_emb`、`actions`、`actions_mask`。

### RoboTwin clean/aug 数据契约

- 数据说明见 `robotwin_clean_aug_lerobot_analysis.md`。
- `robbyant/robotwin-clean-and-aug-lerobot` 顶层包含 `lerobot_robotwin_eef_clean_50/`、`lerobot_robotwin_eef_aug_500/` 和 `empty_emb.pt`。
- 每个任务目录是一个近似独立的 LeRobot v2.1 dataset：`data/*.parquet` 存逐帧 16 维 EEF action/state，`videos/*.mp4` 存三路相机视频，`meta/*.jsonl` 存 episode、task 和统计信息，`latents/*.pth` 存 WAN 2.2 预提取视频 latent 和 text embedding。
- Stage-1 latent 后训练主要读取 `latents/*.pth` 和 parquet 的 `action`。它不直接解码 `videos/*.mp4`。
- latent 文件名必须与 `meta/episodes.jsonl` 的 `action_config` 对齐，例如 `episode_000000_0_139.pth` 对应 `episode_index=0,start_frame=0,end_frame=139`。
- RoboTwin 原始 action 是 16 维 EEF pose + gripper。配置仍保持标准 `action_dim=30`，通过 `inverse_used_action_channel_ids` 把 16 维有效数据放入 30 维布局，并用 `actions_mask` 屏蔽无效通道。

### 配置层

- `wan_va/configs/shared_config.py`：共享 host、port、dtype、save root、patch size、offload 默认值。
- `wan_va/configs/__init__.py`：导入所有具体配置，并注册到 `VA_CONFIGS`。
- `wan_va/configs/va_robotwin_cfg.py`：RoboTwin 推理配置。当前工作区把 `wan22_pretrained_model_name_or_path` 指向 `/mnt/data/share/checkpoints/robbyant/lingbot-va-base`。
- `wan_va/configs/va_libero_cfg.py`、`va_franka_cfg.py`、`va_demo_cfg.py`：分别定义环境图像尺寸、camera key、action 维度、推理步数、通道映射和分位数归一化统计。
- `wan_va/configs/*_train_cfg.py`：继承环境推理配置，并补充 dataset path、`empty_emb_path`、wandb、dataloader worker、checkpoint 间隔、学习率、weight decay、梯度累积和训练 step 数。
- `wan_va/configs/va_robotwin_latent_posttrain_cfg.py`：继承 `robotwin_train`，默认 10000 steps、`latent_batch_size=8`、`latent_learning_rate=1e-4`、`save_root=./train_out_wam_latent`。
- `wan_va/configs/va_libero_latent_posttrain_cfg.py`：继承 `libero_train`，默认 5000 steps、`latent_batch_size=8`、`latent_learning_rate=1e-4`、`save_root=./train_out_wam_latent_libero`。
- `wan_va/configs/*_i2va.py`：继承环境推理配置，并补充初始图片目录、生成 chunk 数、prompt、`infer_mode="i2va"`。

### 工具与分布式

- `wan_va/utils/scheduler.py`：`FlowMatchScheduler` 负责 shifted sigma/timestep schedule、加噪、flow target、推理 step 和训练权重。
- `wan_va/utils/utils.py`：patch 还原、rotary grid id 生成、异步保存、timestep 采样、warmup lambda。
- `wan_va/utils/sever_utils.py`：websocket 服务编排，以及 rank 0 到 worker rank 的分布式广播。
- `wan_va/utils/logging.py`：初始化 root logger，并压低 profiler 日志。
- `wan_va/distributed/fsdp.py`：activation checkpoint wrapper 和按 block 调用 `fully_shard` 的 FSDP 设置。
- `wan_va/distributed/util.py`：模型设备/FSDP 配置、NCCL 分布式初始化、跨 rank 指标归约。

### 远程推理协议

- `wan_va/utils/Simple_Remote_Infer/deploy/websocket_policy_server.py`
  - 服务一个带 `infer(obs)` 方法的 policy 对象。
  - 使用 msgpack/numpy 序列化。
  - 提供 `/healthz` 健康检查。

- `wan_va/utils/Simple_Remote_Infer/deploy/websocket_client_policy.py`
  - 连接服务端并循环等待服务可用。
  - 关闭 ping timeout，适配长时间推理。
  - 发送 observation dict，并解包返回 dict。

## 主流程

### 训练流程

1. `script/run_va_posttrain.sh` 使用类似 `robotwin_train` 的 config name 启动 `wan_va.train`。
2. `Trainer` 初始化分布式环境，加载 transformer，应用 activation checkpointing/FSDP，并创建 optimizer/scheduler。
3. `MultiLatentLeRobotDataset` 产出 latent/action batch。
4. `_prepare_input_dict()` 分别给 latent 和 action 加 flow-matching noise，并构造 grid id。
5. `WanTransformer3DModel.forward_train()` 联合处理 video/action token。
6. `compute_loss()` 把 sequence prediction 还原成 patch tensor，并计算按 frame 加权的 video/action MSE。
7. checkpoint 只保存 transformer，且保持 Diffusers 兼容格式。

### Stage-1 latent 后训练流程

1. `script/run_wam_latent_posttrain_stage1.sh` 检查 `DATASET_PATH`、`empty_emb.pt`、LeRobot `meta/info.json`、`latents/` 和 `PRETRAINED_MODEL/transformer` 后，用 `torch.distributed.run` 启动 `wan_va.train_latent_posttrain`。
2. `Stage1LatentPostTrainer` 加载 frozen transformer，并根据 transformer hidden dim 和 config action dim 创建 `WAMControllableSubspaceProbe`。
3. Dataset 对每个 `action_config` 片段读取三路 camera latent、text embedding 和 parquet action；RoboTwin 下把 absolute EEF action 转 relative pose，再映射到 30 维 action tensor。
4. `_extract_branch_hiddens()` 从 frozen transformer 的 input embedding 或 hidden 中得到 video/action tokens；video 侧默认使用最后一帧 token 减第一帧 token作为 `video_delta_tokens`。
5. Probe 输出 `c_v` 和 `c_a`，并计算 action recon、video-to-action inverse、counterfactual、可选 relational loss。
6. 训练只更新 probe；negative queue 保存历史 `c_a` 作为 counterfactual negatives。
7. checkpoint 写到 `latent_posttrain_stage1`，用于后续 Stage-2 action path consistency 训练或离线分析。

### 服务端推理流程

1. client 发送 `dict(reset=True, prompt=...)` 初始化 prompt embedding、cache、维度、action mask 和归一化统计。
2. 可选发送 `compute_kv_cache=True`，把历史 observation/action 编码并写入 transformer cache。
3. 普通 infer 会先用 VAE 编码当前 observation，然后去噪未来视频 latent，再去噪 action chunk。
4. `postprocess_action()` 反归一化，并只保留当前配置声明的 action 通道。
5. 如果请求可视化，服务端解码预测 latent，并在返回中附带 `video`。

## 重点实现解释

- 视频和 action 在概念上是统一序列，但推理时分两个 denoising loop：先视频，再 action。
- action 的 grid id 会把空间坐标置为 `-1`，并给 frame 加小数偏移，让 action token 有独立的 rotary 位置模式。
- 训练时 `cfg_prob` 会把文本 embedding 替换为 `empty_emb`，为 classifier-free guidance 做准备。
- `action_mask` 会屏蔽标准 30 维布局中未使用的通道，避免它们参与预测或 loss。
- cache 里用 `is_pred` 区分预测 token 和已提交历史；`clear_pred_cache()` 会清理生成预测 cache，但保留观察历史。
- `robotwin_tshape` 同时影响数据集 latent 拼接和服务端 observation 编码/解码，是 RoboTwin 相机布局的特殊路径。

## 需要注意的风险

- `wan_va/distributed/util.py:init_distributed()` 无条件调用 `dist.init_process_group`，所以本地入口默认需要 `torch.distributed.run` 环境。
- 部分配置文件在复制 `va_shared_cfg` 后才修改 `va_shared_cfg.infer_mode`，这依赖导入顺序，不够直观。
- `pyproject.toml` 写的是 `packages = ["lingbot_va"]`，但源码目录是 `wan_va/`；依赖打包/安装前需要验证。
- `LatentLeRobotDataset.__init__()` 的 fallback 分支引用了本文件未导入的名称：`packaging`、`get_safe_version`、`download_videos`。如果本地 LeRobot 文件齐全，常见路径可能不会触发；缺缓存场景需要验证。
