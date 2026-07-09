# 项目

## 概览

LingBot-VA 是一个机器人视频-动作世界模型项目。核心代码在 `wan_va/`：它加载 Wan 风格的视频扩散骨干，将视频 latent 建模和机器人 action 建模合到同一个 transformer 框架里，并提供后训练、websocket 服务端推理、以及 image-to-video-action 离线生成流程。

## 技术栈

- Python 包代码位于 `wan_va/`。
- 主要运行依赖包括 PyTorch、CUDA/NCCL 分布式、Diffusers Wan VAE、Transformers T5/UMT5 文本编码器、LeRobot 数据集、safetensors、websockets、msgpack、einops、numpy、scipy、wandb。
- `README.md` 声明的基准环境是 Python 3.10.16、PyTorch 2.9.0、CUDA 12.6。
- 训练使用 FSDP 和 activation checkpointing。
- 推理使用 server-client 架构，模型服务和仿真客户端通过 websocket 通信，便于隔离依赖环境。

## 入口文件

- `wan_va/train.py`：主 WAM 后训练入口。通过 `VA_CONFIGS` 读取配置，初始化分布式环境，创建 `Trainer`，按 step 训练 transformer。
- `wan_va/train_latent_posttrain.py`：Stage-1 latent 后训练入口。冻结 WAM transformer，只训练 controllable-subspace probe / adapter。
- `wan_va/wan_va_server.py`：推理和 image-to-video-action 生成入口。创建 `VA_Server` 后，根据配置进入 websocket 服务模式或离线 `i2va` 生成模式。
- `script/run_va_posttrain.sh`：通过 `python -m torch.distributed.run -m wan_va.train` 启动训练。
- `script/run_wam_latent_posttrain_stage1.sh`：通过 `python -m torch.distributed.run -m wan_va.train_latent_posttrain` 启动 Stage-1 latent 后训练，并在启动前检查数据目录、`empty_emb.pt`、`meta/info.json`、`latents/` 和 pretrained transformer。
- `script/run_launch_va_server_sync.sh`：通过 `python -m torch.distributed.run -m wan_va.wan_va_server` 启动服务/生成。
- `evaluation/robotwin/launch_server.sh` 与 `evaluation/libero/launch_server.sh`：benchmark 服务端启动脚本。
- `evaluation/robotwin/launch_server_multigpus.sh`：RoboTwin 多 GPU server 启动脚本，按 `START_PORT+i` 启动多个 websocket server。
- `evaluation/robotwin/launch_rt_c2r_benchmark.sh`：RoboTwin RT-C2R 批量评测入口，遍历 6 个 split 和 50 个任务，支持 validated seed manifests、日志目录、自动启动 server 和 summary 输出。
- `evaluation/robotwin/run_rt_c2r_*.sh`：固定 checkpoint 的 RT-C2R wrapper，每任务默认跑 3 个 episode，并把结果写到 `c2r_bench_result/<case>/`。
- `evaluation/robotwin/launch_client.sh` 与 `evaluation/libero/launch_client.sh`：benchmark 客户端启动脚本。

## 重要目录

- `wan_va/configs/`：EasyDict 配置注册表，包含共享默认值、环境推理配置、训练配置和 `i2va` 配置。
- `wan_va/dataset/`：LeRobot 数据适配层，把离线 Wan VAE latent 和 LeRobot metadata 中的 action segment 对齐。
- `wan_va/modules/`：模型定义和加载工具，包括 VAE、tokenizer、text encoder、`WanTransformer3DModel`。
- `wan_va/modules/latent_posttraining.py`：Stage-1 latent 后训练的 probe / adapter、masked loss、counterfactual loss、relational loss 和 retrieval metrics。
- `wan_va/distributed/`：FSDP 分片、activation checkpointing、分布式初始化和分布式指标归约。
- `wan_va/utils/`：scheduler、grid/patch 辅助函数、异步保存、日志和 websocket 服务编排。
- `wan_va/utils/Simple_Remote_Infer/`：轻量 websocket client/server 工具。
- `robotwin_clean_aug_lerobot_analysis.md`：RoboTwin clean/aug LeRobot 数据结构调研，记录 `data/`、`videos/`、`meta/`、`latents/` 的含义和训练使用方式。

## 架构要点

- 训练时模型处理的是视频 latent，不是原始像素视频。数据集中的 `latents/` 路径按 LeRobot `videos/` 路径镜像组织。
- action 使用标准 30 维布局。每个环境通过 `used_action_channel_ids` 选择自身有效通道，并把缺失通道补零。
- 训练同时预测 video latent 和 action 的 flow-matching target，loss 分成 latent loss 与 action loss。
- Stage-1 latent 后训练不更新 transformer。它从冻结 transformer 的 video/action hidden 或 input embedding 中提取表征，只训练一个小型 probe，使 video dynamics 表征和 action 表征在 controllable subspace 上相关。
- RoboTwin latent 后训练使用 `robbyant/robotwin-clean-and-aug-lerobot` 的 LeRobot v2.1 episode-per-file 结构：低维 action 来自 `data/*.parquet`，语言和片段来自 `meta/episodes.jsonl`，视频表征来自预提取 `latents/*.pth`，训练时不直接解码 `videos/*.mp4`。
- 推理先把 observation 编码成 VAE latent，再去噪生成未来视频 latent 和 action chunk，最后把 action 反归一化并裁剪回当前环境需要的通道。
- `wan_va/wan_va_server.py --pretrained-model` 现在支持两类路径：完整模型目录（含 `vae/ tokenizer/ text_encoder/ transformer/`）或训练 checkpoint 目录（含 `transformer/config.json`）。训练 checkpoint 会复用 config 中的 base 模型组件，只替换 transformer。
- RT-C2R benchmark 使用 `evaluation/robotwin/rt_c2r_validated_manifests/*.jsonl` 作为 expert-validated seed 预设，结果汇总到 `summary.json` 和 `summary.csv`。
- `attn_mode` 与运行模式强相关：主 WAM 训练使用 `flex`；推理使用 `torch` 或 `flashattn`；Stage-1 latent 后训练配置默认 `latent_hidden_attn_mode="torch"`。

## 开发命令

已从项目文件确认但本次未执行：

```bash
pip install -r requirements.txt
make format
NGPU=8 CONFIG_NAME='robotwin_train' bash script/run_va_posttrain.sh
NGPU=8 CONFIG_NAME='libero_train' bash script/run_va_posttrain.sh
DATASET_PATH=/path/to/robotwin-clean-and-aug-lerobot NGPU=8 bash script/run_wam_latent_posttrain_stage1.sh
bash evaluation/robotwin/launch_server.sh
bash evaluation/libero/launch_server.sh

# RT-C2R 4 卡 server 示例。换 checkpoint 前需要先停掉旧 server。
GPU_IDS=0,1,2,3 NUM_GPUS=4 START_PORT=29556 MASTER_PORT=29661 \
  bash evaluation/robotwin/launch_server_multigpus.sh \
  /mnt/data/users/xianyi/EmbodyAi/lingbot-va/ckpts/train_out_contrastive_align_clean/checkpoints/checkpoint_step_5000

# 使用外部已启动 server 跑每任务 3 个 episode 的 RT-C2R case。
AUTO_START_SERVER=False NUM_GPUS=4 START_PORT=29556 \
  bash evaluation/robotwin/run_rt_c2r_contrastive_align_clean_step5000.sh
```

## 约定

- 新增运行配置时优先放到 `wan_va/configs/`，并在 `wan_va/configs/__init__.py` 注册到 `VA_CONFIGS`。
- checkpoint 保持 Diffusers 风格目录：`transformer/config.json` 和 `transformer/diffusion_pytorch_model.safetensors`。
- 除非明确修改数据/动作契约，否则保留 30 维标准 action 布局。
- `wan_va/utils/sever_utils.py` 当前就是有效导入路径；虽然文件名疑似拼写错误，改名需要同步更新导入。
