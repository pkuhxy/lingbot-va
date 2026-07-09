# AI 交接

## 快速背景

本仓库是 LingBot-VA，一个机器人视频-动作世界模型项目。主要代码在 `wan_va/`：它用预计算视频 latent 的 LeRobot 数据做 WAM 后训练，并通过 websocket 服务端为 RoboTwin/LIBERO 风格客户端提供 action 推理。当前新增了一条 Stage-1 latent post-training 路径：冻结 pretrained transformer，只训练小型 controllable-subspace probe 来关联 video dynamics 表征和 action 表征。最近又补了 RT-C2R benchmark wrapper、validated manifest 评测路径，以及训练 checkpoint 目录的 server 加载兼容。

## 当前状态
- 最近用户目标：整理 RT-C2R benchmark 的四个 checkpoint 评测脚本、4 卡 server 启动方式、结果目录约定和 server 加载修复。
- 文档最近更新：2026-06-24。
- RT-C2R wrapper 已创建：`evaluation/robotwin/run_rt_c2r_contrastive_align_clean_step5000.sh`、`run_rt_c2r_robotwin_clean_step5000.sh`、`run_rt_c2r_lingbot_va_base.sh`、`run_rt_c2r_lingbot_va_posttrain_robotwin.sh`。
- 默认每任务跑 `TEST_NUM=3`，使用 `evaluation/robotwin/rt_c2r_validated_manifests`，输出到 `c2r_bench_result/<case>/`。
- `wan_va/wan_va_server.py` 已支持传训练 checkpoint 目录：如果没有 `vae/` 但有 `transformer/config.json`，server 使用 base model 的 `vae/tokenizer/text_encoder`，只加载该 checkpoint 的 transformer。
- 最新 server smoke：`logs/server_0_20260624_201301.log` 显示监听 `0.0.0.0:29556` 并有 client 连接；完整四 checkpoint benchmark 尚未确认完成。
- 工作区已有用户/本地改动和运行产物；没有明确指令时不要回退这些改动。

## 优先阅读

- `evaluation/robotwin/launch_rt_c2r_benchmark.sh`：RT-C2R 批量评测入口，负责 split/task 遍历、validated manifest、日志、pid、summary。
- `evaluation/robotwin/run_rt_c2r_*.sh`：四个 checkpoint 的固定 wrapper。
- `evaluation/robotwin/launch_server_multigpus.sh`：4 卡 websocket server 启动脚本，端口为 `START_PORT+i`。
- `wan_va/train_latent_posttrain.py`：Stage-1 latent 后训练流程，冻结 transformer，只训练 probe。
- `wan_va/modules/latent_posttraining.py`：`WAMControllableSubspaceProbe`、masked loss、counterfactual / relational loss、retrieval metrics。
- `wan_va/configs/va_robotwin_latent_posttrain_cfg.py`：RoboTwin latent posttrain 参数。
- `script/run_wam_latent_posttrain_stage1.sh`：Stage-1 启动脚本和数据目录预检查。
- `robotwin_clean_aug_lerobot_analysis.md`：RoboTwin clean/aug 数据结构和训练使用说明。
- `wan_va/train.py`：训练流程和 loss。
- `wan_va/wan_va_server.py`：服务端推理和 `i2va` 生成。
- `wan_va/modules/model.py`：video/action transformer 实现。
- `wan_va/dataset/lerobot_latent_dataset.py`：LeRobot latent/action 数据契约。
- `wan_va/configs/__init__.py`：可用 config name。
- `wan_va/configs/va_robotwin_cfg.py`：当前本地 RoboTwin 路径、action/camera 配置。
- `wan_va/utils/Simple_Remote_Infer/deploy/websocket_client_policy.py`：client 协议。
- `wan_va/utils/Simple_Remote_Infer/deploy/websocket_policy_server.py`：server 协议。

## 命令

这些命令来自仓库文件，本次未执行：

```bash
pip install -r requirements.txt
make format
NGPU=8 CONFIG_NAME='robotwin_train' bash script/run_va_posttrain.sh
DATASET_PATH=/path/to/robotwin-clean-and-aug-lerobot NGPU=8 bash script/run_wam_latent_posttrain_stage1.sh
DATASET_PATH=/path/to/robotwin-clean-and-aug-lerobot NUM_STEPS=2 NGPU=1 bash script/run_wam_latent_posttrain_stage1.sh
NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh
bash evaluation/robotwin/launch_server.sh
bash evaluation/robotwin/launch_client.sh results adjust_bottle

# 4 卡启动本地训练 checkpoint server。换 checkpoint 前要停旧 server。
GPU_IDS=0,1,2,3 NUM_GPUS=4 START_PORT=29556 MASTER_PORT=29661 \
  bash evaluation/robotwin/launch_server_multigpus.sh \
  /mnt/data/users/xianyi/EmbodyAi/lingbot-va/ckpts/train_out_contrastive_align_clean/checkpoints/checkpoint_step_5000

# 用外部已启动 server 跑对应 RT-C2R case。
AUTO_START_SERVER=False NUM_GPUS=4 START_PORT=29556 \
  bash evaluation/robotwin/run_rt_c2r_contrastive_align_clean_step5000.sh
```

## 约束与约定

- 保留 dirty worktree 中的用户改动。
- 除非修改 `init_distributed()`，否则入口脚本优先通过 `torch.distributed.run` 启动。
- 主 WAM 训练需要 transformer config `attn_mode="flex"`；推理需要 `torch` 或 `flashattn`；Stage-1 latent posttrain 默认 `latent_hidden_attn_mode="torch"`。
- action tensor 保持 30 维标准布局，使用 `used_action_channel_ids` 和 `inverse_used_action_channel_ids` 做映射。
- RoboTwin raw action 是 16 维 EEF pose + gripper；latent posttrain dataset 会转 relative pose、映射进 30 维标准 action，并用 `actions_mask` 屏蔽无效通道。
- Stage-1 latent posttrain 的 `DATASET_PATH` 应指向包含 `empty_emb.pt` 的数据集顶层，通常是 `robotwin-clean-and-aug-lerobot/`，不要只指向 clean/aug 子目录或单个任务目录。
- Stage-1 训练使用 `latents/*.pth` 和 parquet action，不直接解码 `videos/*.mp4`。
- RT-C2R wrapper 默认 `AUTO_START_SERVER=True`；如果用户已经手动启动多卡 server，运行 wrapper 时必须传 `AUTO_START_SERVER=False NUM_GPUS=4 START_PORT=29556`。
- 外部 server 加载的是哪个 checkpoint，eval 结果就对应哪个 checkpoint；四个 case 需要逐个重启 server 后运行对应 wrapper。
- `wan_va/utils/sever_utils.py` 文件名疑似拼写错误，但当前是有效导入路径。

## 风险与开放问题

- packaging 可能有问题：`pyproject.toml` 引用 `lingbot_va`，不是 `wan_va`。
- dataset 缺缓存 fallback 路径引用未导入名称，需要用真实缺缓存场景验证。
- config 应检查并显式设置 `infer_mode`。
- 当前 RoboTwin checkpoint 路径是本机路径。
- Stage-1 DataLoader 默认 collate 要求 batch 内 latent/action shape 一致；真实数据如果有不同 latent 帧数，需要固定窗口采样或 padding collate。
- Stage-1 checkpoint 只是 probe，不是可直接部署的 transformer；后续 Stage-2 还需要接入 latent consistency loss。
- RT-C2R 完整 benchmark 尚未完成确认；需要检查每个 case 的 `summary.json`、`summary.csv` 和 client logs。
