# 状态

## 当前快照

- Last updated: 2026-06-15
- Branch: `main...origin/main`
- 本次更新 `vibe-doc/`，补充新增 WAM Stage-1 latent 后训练代码、启动脚本和 RoboTwin clean/aug LeRobot 数据说明。
- 当前用户目标：让未来 agent 能快速理解 `wan_va/train_latent_posttrain.py`、`wan_va/modules/latent_posttraining.py`、latent posttrain config、启动脚本和数据处理方式。
- 写文档前工作区已经存在本地未提交改动。

## 已完成

- 新增 `vibe-doc/PROJECT.md`，记录项目级稳定上下文。
- 新增 `vibe-doc/ARCHITECTURE.md`，集中解释 `wan_va/` 代码结构、主流程和重点实现。
- 新增 `STATUS.md`、`DECISIONS.md`、`TASKS.md`、`AI_HANDOFF.md`，方便未来 agent 接手。
- 已补充 Stage-1 latent 后训练文档：
  - `wan_va/train_latent_posttrain.py`：冻结 transformer、训练 `WAMControllableSubspaceProbe`。
  - `wan_va/modules/latent_posttraining.py`：probe / adapter、masked loss、counterfactual、relational、retrieval metrics。
  - `wan_va/configs/va_robotwin_latent_posttrain_cfg.py` 与 `wan_va/configs/va_libero_latent_posttrain_cfg.py`：latent posttrain config。
  - `script/run_wam_latent_posttrain_stage1.sh`：分布式启动脚本和数据目录预检查。
  - `robotwin_clean_aug_lerobot_analysis.md`：RoboTwin clean/aug 数据结构说明。

## 观察到的本地代码状态

- `wan_va/configs/va_robotwin_cfg.py` 有本地改动：`wan22_pretrained_model_name_or_path` 指向 `/mnt/data/share/checkpoints/robbyant/lingbot-va-base`。
- `wan_va/wan_va_server.py` 有本地改动：在 `VA_Server.__init__()` 初始化 `VideoProcessor`，并在 `infer()` 中支持按请求返回解码视频。
- `evaluation/robotwin/eval_polict_client_openpi.py` 和 `evaluation/robotwin/launch_client.sh` 有本地改动，涉及 RoboTwin root 查找、websocket host、可视化输出、ffmpeg 处理和 debug 数据保存。
- 新增/未跟踪的 latent 后训练相关文件包括 `wan_va/train_latent_posttrain.py`、`wan_va/modules/latent_posttraining.py`、`wan_va/configs/va_robotwin_latent_posttrain_cfg.py`、`wan_va/configs/va_libero_latent_posttrain_cfg.py`、`script/run_wam_latent_posttrain_stage1.sh`、`robotwin_clean_aug_lerobot_analysis.md`、`handoff_wam_latent_posttraining.md`。
- 当前存在未跟踪运行产物目录/文件：`logs/`、`results/`、`results_base/`、`visualization/`、多个 `pids_all_tasks_*.txt`。

## 验证

- 已用 `find`、`grep`、`sed`、`wc`、`git status`、定向 `git diff` 做源码检查。
- 已执行 `bash -n script/run_wam_latent_posttrain_stage1.sh`，脚本语法检查通过。
- 未运行 Stage-1 训练、推理、格式化、import check 或真实 dataset 构造测试。

## 已知问题

- 当前环境没有 `rg`，文件检索改用 `find`。
- `wan_va/distributed/util.py:init_distributed()` 无条件初始化 NCCL 分布式环境；除非修改该函数，否则建议通过脚本启动。
- `pyproject.toml` 疑似打包名不一致：声明 `lingbot_va`，但源码目录是 `wan_va/`。
- `wan_va/dataset/lerobot_latent_dataset.py` 的缺缓存 fallback 路径可能失败，因为引用了本文件未导入的名称。
- 部分 config 依赖 shared config 的导入/修改顺序来得到 `infer_mode`，显式写到每个 config 会更清楚。
- Stage-1 latent 后训练 DataLoader 使用默认 collate，实际训练假设 batch 内 sample 的 latent/action shape 一致；如果真实数据存在不同 latent 帧数或空间尺寸，需要固定窗口采样或 padding collate。
- `script/run_wam_latent_posttrain_stage1.sh` 要求 `DATASET_PATH` 指向包含 `empty_emb.pt` 的数据集顶层；只指向 `lerobot_robotwin_eef_clean_50/` 或单个任务目录会失败。
