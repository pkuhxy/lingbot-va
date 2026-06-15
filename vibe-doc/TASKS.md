# 任务

## 下一步

- [ ] 用真实 `robbyant/robotwin-clean-and-aug-lerobot` 路径跑 Stage-1 dataset 构造 smoke test，确认能发现所有任务目录、`new_metas` 非空、单条 sample 包含 `latents/text_emb/actions/actions_mask`。
- [ ] 检查真实 batch 中 `latents`、`actions`、`actions_mask` shape 是否可被默认 DataLoader collate；如果 episode latent 长度不一致，补固定窗口采样或 padding collate。
- [ ] 用小步数运行 Stage-1 smoke test：`DATASET_PATH=/path/to/robotwin-clean-and-aug-lerobot NUM_STEPS=2 NGPU=1 bash script/run_wam_latent_posttrain_stage1.sh`。
- [ ] 验证 Stage-1 checkpoint：确认 `<save_root>/latent_posttrain_stage1/checkpoint_step_*/stage1_probe.pt` 可加载，且包含 `probe`、`optimizer`、`step`、`config`。
- [ ] 验证包安装：`pyproject.toml` 声明 `lingbot_va`，但源码目录是 `wan_va/`。
- [ ] 在目标环境中跑轻量 import check：`wan_va.configs`、`wan_va.modules.model`、`wan_va.dataset`。
- [ ] 检查每个 config 是否都显式拥有 `infer_mode`，避免依赖 `va_shared_cfg` 的修改顺序。
- [ ] 决定是否把 `wan_va/utils/sever_utils.py` 改名为 `server_utils.py`，并同步更新导入。
- [ ] 确认 `wan_va/configs/va_robotwin_cfg.py` 中的本机 RoboTwin checkpoint 路径是否应该提交，或改成 override/环境变量。

## 仍需验证

- [ ] 只有在实际修改代码文件后才运行 `make format`。
- [ ] 跑一个最小 distributed launch smoke test，确认服务端能启动。
- [ ] 用真实 LeRobot dataset path 和 latent 文件跑主训练 dataset 构造 smoke test。
- [ ] 用 fake 或录制 observation 数据跑一个 websocket client/server round trip。
- [ ] 对 `script/run_wam_latent_posttrain_stage1.sh` 做一次真实启动前检查，确认默认 `PRETRAINED_MODEL=/mnt/data/share/checkpoints/robbyant/lingbot-va-base` 在目标机器上存在。

## 开放问题

- [ ] `enable_offload` 是否应该像 `shared_config.py` 一样默认 `False`，还是按 README 的显存说明默认开启 VAE/text encoder offload？
- [ ] `logs/`、`results/`、`results_base/`、`visualization/`、`pids_all_tasks_*.txt` 是否都只是本地运行产物？
- [ ] `script/run_va_posttrain.sh` 中的 wandb placeholder 是否应该改为完全由外部环境变量提供？
- [ ] Stage-1 后训练的 `latent_hidden_source` 默认用 `embed`，是否需要在正式实验中对比 `transformer` hidden。
- [ ] `lambda_relational` 当前默认 0，是否作为 ablation 打开。

## 暂缓

- [ ] 如果文档用于新人 onboarding，可补一个架构图或时序图。
- [ ] 为 action channel mapping 和 latent/action shape contract 增加测试。
- [ ] 为 `WanTransformer3DModel.forward_train()`、`VA_Server._infer()` 和 dataset action 预处理补更清晰的 docstring。
- [ ] 实现 Stage-2：加载 Stage-1 probe，冻结或半冻结 video branch，对 action path 加 latent consistency loss。
