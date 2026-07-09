# 状态

## 当前快照

- Last updated: 2026-06-24
- Branch: `main`，工作区有本地未提交改动和运行产物。
- 当前用户目标：整理 RT-C2R benchmark 准备工作、四个 checkpoint 的评测脚本、4 卡 server 启动方式，以及本地训练 checkpoint 加载修复。
- 重要现状：`launch_server_multigpus.sh` 启动本地 `checkpoint_step_5000` 时曾因缺少 `vae/config.json` 失败；已修改 `wan_va/wan_va_server.py`，让训练 checkpoint 目录只替换 `transformer/`，其余组件继续来自 base model。
- 最新 `logs/server_0_20260624_201301.log` 显示 server 已监听 `0.0.0.0:29556`，并有 client 连接和 reset 记录；完整 RT-C2R benchmark 结果仍未确认完成。

## RT-C2R 相关实现

- 新增四个 wrapper：
  - `evaluation/robotwin/run_rt_c2r_contrastive_align_clean_step5000.sh`
  - `evaluation/robotwin/run_rt_c2r_robotwin_clean_step5000.sh`
  - `evaluation/robotwin/run_rt_c2r_lingbot_va_base.sh`
  - `evaluation/robotwin/run_rt_c2r_lingbot_va_posttrain_robotwin.sh`
- 每个 wrapper 默认 `TEST_NUM=3`，使用 `evaluation/robotwin/rt_c2r_validated_manifests`，输出到 `c2r_bench_result/<case>/`。
- `launch_rt_c2r_benchmark.sh` 支持 `AUTO_START_SERVER=True/False`、可配置 `LOG_DIR/PID_DIR/SERVER_LOG_DIR/SERVER_SAVE_ROOT`、绝对 `POLICY_CONFIG`，并输出 `summary.json` 和 `summary.csv`。
- `eval_polict_client_openpi.py` 的辅助 `eval_result/` 已改为写入当前 `save_root` 下，避免散落到仓库根目录。
- `launch_server.sh` 和 `launch_server_multigpus.sh` 会定位并切换到项目根目录后再启动，降低从其他目录调用时的路径风险。

## 运行与日志观察

- 本地训练 checkpoint 目录结构只包含 `transformer/`、`align_head/` 和 `trainer_state.pt`；不是完整 Diffusers model root。
- 完整模型目录 `/mnt/data/share/checkpoints/robbyant/lingbot-va-base` 与 `/mnt/data/share/checkpoints/robbyant/lingbot-va-posttrain-robotwin` 包含 `vae/ tokenizer/ text_encoder/ transformer/`。
- `logs/server_0_20260624_195012.log` 记录了旧失败：server 试图读取 `checkpoint_step_5000/vae/config.json`。
- `logs/server_0_20260624_201301.log` 记录了修复后的 server 启动成功，监听端口 `29556`。
- `c2r_bench_result/contrastive_align_clean_step5000/client_logs/easy_0_stack_bowls_three_20260624_195031.log` 记录了旧 server 未就绪时的 `ConnectionRefusedError` / `KeyboardInterrupt`。

## 验证

- 已执行并通过：`bash -n evaluation/robotwin/launch_rt_c2r_benchmark.sh evaluation/robotwin/launch_server.sh evaluation/robotwin/launch_server_multigpus.sh evaluation/robotwin/run_rt_c2r_*.sh`。
- 已执行并通过：`python -m py_compile evaluation/robotwin/eval_polict_client_openpi.py evaluation/robotwin/calc_rt_c2r_stat.py wan_va/wan_va_server.py`。
- 已确认四个 checkpoint 路径、`evaluation/robotwin/rt_c2r_validated_manifests/rt_c2r_*.jsonl`、`RoboTwin/policy/ACT/deploy_policy.yml` 存在。
- 未确认：四个 checkpoint 的完整 6 split x 50 tasks x 3 episodes RT-C2R benchmark 是否都已跑完并产出 `summary.json`。

## 观察到的本地代码状态

- `wan_va/configs/va_robotwin_cfg.py` 有本地改动：`wan22_pretrained_model_name_or_path` 指向 `/mnt/data/share/checkpoints/robbyant/lingbot-va-base`。
- `wan_va/wan_va_server.py` 有本地改动：支持请求可视化返回解码视频；并支持 `--pretrained-model` 传训练 checkpoint 目录，只替换 `transformer/`。
- `evaluation/robotwin/eval_polict_client_openpi.py`、`evaluation/robotwin/launch_rt_c2r_benchmark.sh`、`evaluation/robotwin/launch_server.sh`、`evaluation/robotwin/launch_server_multigpus.sh` 有 RT-C2R 相关本地改动。
- 新增/未跟踪的 latent 后训练相关文件包括 `wan_va/train_latent_posttrain.py`、`wan_va/modules/latent_posttraining.py`、`wan_va/configs/va_robotwin_latent_posttrain_cfg.py`、`wan_va/configs/va_libero_latent_posttrain_cfg.py`、`script/run_wam_latent_posttrain_stage1.sh`、`robotwin_clean_aug_lerobot_analysis.md`、`handoff_wam_latent_posttraining.md`。
- 当前存在未跟踪运行产物目录/文件：`logs/`、`results/`、`results_base/`、`visualization/`、`c2r_bench_result/`、多个 `pids_all_tasks_*.txt` 和 `pids_rt_c2r_*.txt`。

## 既有已知问题

- `wan_va/distributed/util.py:init_distributed()` 无条件初始化 NCCL 分布式环境；除非修改该函数，否则建议通过脚本启动。
- `pyproject.toml` 疑似打包名不一致：声明 `lingbot_va`，但源码目录是 `wan_va/`。
- `wan_va/dataset/lerobot_latent_dataset.py` 的缺缓存 fallback 路径可能失败，因为引用了本文件未导入的名称。
- Stage-1 latent 后训练 DataLoader 使用默认 collate，实际训练假设 batch 内 sample 的 latent/action shape 一致；如果真实数据存在不同 latent 帧数或空间尺寸，需要固定窗口采样或 padding collate。
- `script/run_wam_latent_posttrain_stage1.sh` 要求 `DATASET_PATH` 指向包含 `empty_emb.pt` 的数据集顶层；只指向 `lerobot_robotwin_eef_clean_50/` 或单个任务目录会失败。
