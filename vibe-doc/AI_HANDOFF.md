# AI 交接

## 快速背景

本仓库是 LingBot-VA，一个机器人视频-动作世界模型项目。主要代码在 `wan_va/`：它用预计算视频 latent 的 LeRobot 数据做 WAM 后训练，并通过 websocket 服务端为 RoboTwin/LIBERO 风格客户端提供 action 推理。当前新增了一条 Stage-1 latent post-training 路径：冻结 pretrained transformer，只训练小型 controllable-subspace probe 来关联 video dynamics 表征和 action 表征。

## 当前状态

- 最近用户目标：总结新增的 WAM Stage-1 latent 后训练代码、启动脚本和 RoboTwin clean/aug LeRobot 数据说明。
- 文档最近更新：2026-06-15。
- 先读 `vibe-doc/ARCHITECTURE.md`，那里是 `wan_va/` 的重点代码讲解。
- 本次文档更新没有运行训练或推理；已对 `script/run_wam_latent_posttrain_stage1.sh` 执行 `bash -n`，语法检查通过。
- 工作区已有用户/本地改动；没有明确指令时不要回退这些改动。

## 优先阅读

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
```

## 约束与约定

- 保留 dirty worktree 中的用户改动。
- 除非修改 `init_distributed()`，否则入口脚本优先通过 `torch.distributed.run` 启动。
- 主 WAM 训练需要 transformer config `attn_mode="flex"`；推理需要 `torch` 或 `flashattn`；Stage-1 latent posttrain 默认 `latent_hidden_attn_mode="torch"`。
- action tensor 保持 30 维标准布局，使用 `used_action_channel_ids` 和 `inverse_used_action_channel_ids` 做映射。
- RoboTwin raw action 是 16 维 EEF pose + gripper；latent posttrain dataset 会转 relative pose、映射进 30 维标准 action，并用 `actions_mask` 屏蔽无效通道。
- Stage-1 latent posttrain 的 `DATASET_PATH` 应指向包含 `empty_emb.pt` 的数据集顶层，通常是 `robotwin-clean-and-aug-lerobot/`，不要只指向 clean/aug 子目录或单个任务目录。
- Stage-1 训练使用 `latents/*.pth` 和 parquet action，不直接解码 `videos/*.mp4`。
- `wan_va/utils/sever_utils.py` 文件名疑似拼写错误，但当前是有效导入路径。

## 风险与开放问题

- packaging 可能有问题：`pyproject.toml` 引用 `lingbot_va`，不是 `wan_va`。
- dataset 缺缓存 fallback 路径引用未导入名称，需要用真实缺缓存场景验证。
- config 应检查并显式设置 `infer_mode`。
- 当前 RoboTwin checkpoint 路径是本机路径。
- Stage-1 DataLoader 默认 collate 要求 batch 内 latent/action shape 一致；真实数据如果有不同 latent 帧数，需要固定窗口采样或 padding collate。
- Stage-1 checkpoint 只是 probe，不是可直接部署的 transformer；后续 Stage-2 还需要接入 latent consistency loss。
