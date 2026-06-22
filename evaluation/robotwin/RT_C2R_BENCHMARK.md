# RoboTwin RT-C2R Benchmark

RT-C2R 是一个 clean-to-randomized 评测协议：

```text
RoboTwin clean demonstrations for training -> randomized RoboTwin rollouts for evaluation
```

本仓库侧已经提供 50 个 RoboTwin 任务、6 个 split、每任务每 split 100 个固定 seed 的 manifest。实际 rollout 仍然需要一个可运行的 RoboTwin checkout 和 LingBot-VA 推理 server。

## Benchmark 定义

任务集合：沿用 `evaluation/robotwin/all_tasks.sh` 的 50 个 RoboTwin 双臂任务。

训练集：只用 `lerobot_robotwin_eef_clean_50`，即 50 tasks x 50 clean demos。不要把 `lerobot_robotwin_eef_aug_500` 混入主实验；它只能作为 clean+random upper bound。

评测 split：

| Split | Randomization |
| --- | --- |
| `easy` | no randomization |
| `background` | random background/table texture only |
| `light` | random light only |
| `clutter` | cluttered table only |
| `height` | random table height only |
| `hard` | background + light + clutter + height |

每个 split 默认 50 tasks x 100 seeds = 5,000 episodes；全部 6 个 split 共 30,000 episodes。

## 需要下载的数据和依赖

1. RoboTwin 2.0 simulator 和 assets：

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
git checkout 2eeec322
bash script/_install.sh
bash script/_download_assets.sh
```

2. LingBot-VA base 或待评测 checkpoint：

```bash
huggingface-cli download robbyant/lingbot-va-base --local-dir /path/to/lingbot-va-base
```

3. Clean-only 训练数据：

```bash
huggingface-cli download \
  --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot \
  --local-dir /path/to/robotwin-clean-and-aug-lerobot
```

主实验只使用：

```text
/path/to/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50
```

`empty_emb.pt` 在数据集顶层，所以覆盖训练数据路径时也要显式传 `EMPTY_EMB_PATH`。

## 涉及的代码文件

- `wan_va/configs/va_robotwin_train_cfg.py`：默认已经指向 clean-only 子目录，是 RT-C2R 主实验训练配置。
- `evaluation/robotwin/eval_polict_client_openpi.py`：新增 `seed_manifest` 支持，评测时可以严格复用固定 seeds，并输出 per-episode JSONL。
- `evaluation/robotwin/rt_c2r.py`：50 个 task 和 6 个 split 的 canonical 定义。
- `evaluation/robotwin/generate_rt_c2r_benchmark.py`：生成候选 seed manifests，并可从 RoboTwin `demo_clean.yml` 派生 6 个 `task_config` YAML。
- `evaluation/robotwin/generate_rt_c2r_validated_manifests.py`：用 RoboTwin scripted expert / CuRobo planner / `check_success()` 过滤候选 seeds，只保留 expert 可解 case。
- `evaluation/robotwin/launch_rt_c2r_benchmark.sh`：跨 6 个 split 和 50 个 task 批量评测。
- `evaluation/robotwin/calc_rt_c2r_stat.py`：汇总 split success rate、per-task success rate、Hard/Easy retention 和 Easy-Hard drop。
- `evaluation/robotwin/build_clean_train_manifest.py`：可选，扫描 clean LeRobot 数据并生成训练 manifest。
- `evaluation/robotwin/rt_c2r_manifests/*.jsonl`：已生成的固定评测 seeds。

## 构建 benchmark

已在仓库中生成：

```text
evaluation/robotwin/rt_c2r_manifests/
  rt_c2r_easy.jsonl
  rt_c2r_background.jsonl
  rt_c2r_light.jsonl
  rt_c2r_clutter.jsonl
  rt_c2r_height.jsonl
  rt_c2r_hard.jsonl
```

如果需要重新生成：

```bash
python evaluation/robotwin/generate_rt_c2r_benchmark.py \
  --manifest-dir evaluation/robotwin/rt_c2r_manifests \
  --episodes-per-task 100
```

正式 benchmark 建议生成 expert-validated manifests：

```bash
ROBOTWIN_ROOT=/path/to/RoboTwin \
python evaluation/robotwin/generate_rt_c2r_validated_manifests.py \
  --episodes-per-task 100 \
  --max-candidates-per-task 2000 \
  --num-workers 4 \
  --write-task-configs
```

输出目录：

```text
evaluation/robotwin/rt_c2r_validated_manifests/
  rt_c2r_easy.jsonl
  rt_c2r_background.jsonl
  rt_c2r_light.jsonl
  rt_c2r_clutter.jsonl
  rt_c2r_height.jsonl
  rt_c2r_hard.jsonl
```

每条保留记录都满足 RoboTwin 自带 scripted expert + CuRobo planner 能跑通，且 `check_success()` 通过。被过滤掉的候选 seed 会写到 `evaluation/robotwin/rt_c2r_validation_failures/`，便于排查。

如果想看被过滤 seed 的完整异常堆栈，额外加 `--verbose-failures`。默认不打印 traceback，因为类似 `target_pose cannot be None` 的错误通常只表示该候选 case 对 scripted expert 不合理，应被过滤。

验证脚本默认支持断点续跑：

```bash
# 默认就是 --resume
python evaluation/robotwin/generate_rt_c2r_validated_manifests.py ...

# 对选中的 split/task 从头重跑
python evaluation/robotwin/generate_rt_c2r_validated_manifests.py ... --overwrite

# 完全不读取旧输出和失败日志
python evaluation/robotwin/generate_rt_c2r_validated_manifests.py ... --no-resume
```

断点文件按 task/split 存在：

```text
evaluation/robotwin/rt_c2r_validated_manifests/_tmp_by_task/
```

最终的 `rt_c2r_*.jsonl` 会由这些 task 级断点文件稳定合并生成。失败日志也会参与 resume，用于跳过已经验证失败的 candidate ids。

在 RoboTwin 环境中生成 simulator 可读取的 6 个 task_config：

```bash
ROBOTWIN_ROOT=/path/to/RoboTwin \
python evaluation/robotwin/generate_rt_c2r_benchmark.py \
  --manifest-dir evaluation/robotwin/rt_c2r_manifests \
  --episodes-per-task 100 \
  --write-task-configs \
  --robotwin-root /path/to/RoboTwin
```

这会写入：

```text
/path/to/RoboTwin/task_config/rt_c2r_easy.yml
/path/to/RoboTwin/task_config/rt_c2r_background.yml
/path/to/RoboTwin/task_config/rt_c2r_light.yml
/path/to/RoboTwin/task_config/rt_c2r_clutter.yml
/path/to/RoboTwin/task_config/rt_c2r_height.yml
/path/to/RoboTwin/task_config/rt_c2r_hard.yml
```

## Clean-only 训练

```bash
DATASET_ROOT=/path/to/robotwin-clean-and-aug-lerobot
PRETRAINED_MODEL=/path/to/lingbot-va-base

DATASET_PATH="${DATASET_ROOT}/lerobot_robotwin_eef_clean_50" \
EMPTY_EMB_PATH="${DATASET_ROOT}/empty_emb.pt" \
PRETRAINED_MODEL="${PRETRAINED_MODEL}" \
SAVE_ROOT="./ckpts/robotwin_rt_c2r_clean_only" \
NGPU=8 \
bash script/run_va_posttrain.sh
```

可选生成训练 manifest：

```bash
python evaluation/robotwin/build_clean_train_manifest.py \
  --clean-dataset-root "${DATASET_ROOT}/lerobot_robotwin_eef_clean_50" \
  --output train_manifests/robotwin_clean_50tasks_50demos_per_task.jsonl
```

## 运行 RT-C2R 评测

先启动 LingBot-VA server：

```bash
ROBOTWIN_ROOT=/path/to/RoboTwin bash evaluation/robotwin/launch_server.sh
```

然后运行 benchmark：

```bash
ROBOTWIN_ROOT=/path/to/RoboTwin \
NUM_GPUS=1 \
SAVE_VISUALIZATION=False \
bash evaluation/robotwin/launch_rt_c2r_benchmark.sh ./results/rt_c2r 0 100
```

第三个参数是每任务每 split 的 episode 数。正式报告建议 `100`；快速 pilot 可以用 `20`。

汇总结果：

```bash
python evaluation/robotwin/calc_rt_c2r_stat.py ./results/rt_c2r \
  --output ./results/rt_c2r/summary.json \
  --csv-output ./results/rt_c2r/summary.csv
```

主表报告：

```text
Easy, Background, Light, Clutter, Height, Hard, Hard/Easy Retention
```

同时保留 per-task 表，用于定位具体随机化轴上的退化。

## 注意事项

- 不要用 randomized split 选择 checkpoint；checkpoint selection 只能基于固定 step 或 clean validation。
- launcher 会优先使用 `evaluation/robotwin/rt_c2r_validated_manifests/`；如果该目录存在，默认 `STRICT_SEED_MANIFEST=True`。
- 如果还没有 validated manifests，launcher 会回退到候选 manifests，并默认 `STRICT_SEED_MANIFEST=False`，用于 pilot 和先跑通环境。
- 正式 benchmark 应使用 validated manifests，这样各模型共用同一批 expert-valid seeds。
- LingBot-VA original 的 clean+aug 训练结果可以作为 upper bound，但不要把它称作 clean-to-randomized OOD 主实验。
