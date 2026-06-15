# robbyant/robotwin-clean-and-aug-lerobot 数据格式调研

调研对象：[robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot)

## 结论概览

`robbyant/robotwin-clean-and-aug-lerobot` 不是一个单一表格数据集，而是一个嵌套组织的 **LeRobot v2.1 机器人轨迹数据仓库**。

它将 RoboTwin 双臂操作任务拆成多个任务目录。每个任务目录本身都近似一个独立的 LeRobot dataset：

- 低维状态和动作存放在 `data/*.parquet`
- 多相机视频存放在 `videos/*.mp4`
- 语言任务、episode 信息、统计量和 schema 存放在 `meta/*.json` / `meta/*.jsonl`
- 额外预提取的视频 latent 存放在 `latents/*.pth`

一句话概括：

> 这是 RoboTwin 双臂操作轨迹，按任务拆分为 LeRobot v2.1 episode-per-file 格式；低维 EEF state/action 存 parquet，三路 480x640@50fps 视频存 MP4，语言任务和统计信息存 JSON/JSONL，并额外附带 WAN 2.2 `.pth` 视频 latent。

## 顶层结构

仓库顶层主要有两套数据：

```text
robotwin-clean-and-aug-lerobot/
  lerobot_robotwin_eef_clean_50/
  lerobot_robotwin_eef_aug_500/
  README.md
  empty_emb.pt
```

两套数据含义：

| 目录 | 含义 | 规模 |
| --- | --- | --- |
| `lerobot_robotwin_eef_clean_50` | clean demonstration 数据 | 约 50 个任务，每任务 50 个 episode |
| `lerobot_robotwin_eef_aug_500` | augmented/randomized 数据 | 约 50 个任务，每任务 500 个 episode |

因此总 episode 数约为：

```text
50 tasks x 50 clean episodes + 50 tasks x 500 aug episodes = 27,500 episodes
```

Hugging Face Dataset Viewer 显示约 `82.5k rows`，但这里更像是 HF 自动把所有视频文件展开成预览：

```text
50 tasks x (50 + 500 episodes) x 3 cameras = 82,500 videos
```

所以这个 `82.5k rows` 不等同于 LeRobot 训练时的逐帧样本数。

## 单个任务目录结构

以 clean 数据中的 `adjust_bottle-demo_clean_collect_200-50` 为例：

```text
adjust_bottle-demo_clean_collect_200-50/
  data/
    chunk-000/
      episode_000000.parquet
      episode_000001.parquet
      ...
      episode_000049.parquet

  videos/
    chunk-000/
      observation.images.cam_high/
        episode_000000.mp4
        ...
      observation.images.cam_left_wrist/
        episode_000000.mp4
        ...
      observation.images.cam_right_wrist/
        episode_000000.mp4
        ...

  latents/
    chunk-000/
      observation.images.cam_high/
        episode_000000_0_139.pth
        ...
      observation.images.cam_left_wrist/
      observation.images.cam_right_wrist/

  meta/
    info.json
    episodes.jsonl
    episodes_ori.jsonl
    episodes_stats.jsonl
    tasks.jsonl
```

其中：

- `data/`：逐 episode 的低维轨迹数据，每个 episode 一个 parquet。
- `videos/`：逐 episode 的视频，每个 episode 每个 camera 一个 MP4。
- `meta/`：LeRobot 元数据，包括 schema、语言任务、episode 长度和统计量。
- `latents/`：该仓库额外提供的视频 latent，不是 LeRobot 标准必需目录。

## `meta/info.json`

单个任务目录的 `meta/info.json` 标明这是 LeRobot v2.1 格式：

```json
{
  "codebase_version": "v2.1",
  "robot_type": "aloha",
  "total_episodes": 50,
  "total_frames": 7099,
  "total_tasks": 50,
  "total_videos": 150,
  "total_chunks": 1,
  "chunks_size": 1000,
  "fps": 50,
  "splits": {
    "train": "0:50"
  },
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
}
```

对于 clean 单任务样例：

- `total_episodes = 50`
- `total_videos = 150`，因为每个 episode 有 3 路相机视频
- `fps = 50`

对于 aug 单任务样例：

- `total_episodes = 500`
- `total_videos = 1500`
- `fps = 50`

## Parquet 数据字段

每个 `episode_XXXXXX.parquet` 是一个逐帧表格，不直接存图像像素。

抽样检查 `episode_000000.parquet`，schema 为：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `observation.state` | `list[float32]`, 长度 16 | 当前机器人状态 |
| `action` | `list[float32]`, 长度 16 | 当前帧对应动作 |
| `timestamp` | `float32` | 时间戳，单位秒 |
| `frame_index` | `int64` | 当前 episode 内帧号 |
| `episode_index` | `int64` | episode 编号 |
| `index` | `int64` | 全局帧索引 |
| `task_index` | `int64` | 语言任务索引 |

`observation.state` 和 `action` 都是 16 维，维度名称如下：

```text
left_x
left_y
left_z
left_q1
left_q2
left_q3
left_q4
left_gripper
right_x
right_y
right_z
right_q1
right_q2
right_q3
right_q4
right_gripper
```

这说明该数据使用的是 end-effector pose + gripper 表示，也就是目录名中的 `eef`。它不是普通 14 维关节角 action。

## 视频数据

每个 episode 有三路相机视频：

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
```

在 `info.json` 中，这三个字段都是 `dtype: video`：

| 字段 | shape | 编码 | fps | 音频 |
| --- | --- | --- | --- | --- |
| `observation.images.cam_high` | `[3, 480, 640]` | AV1 | 50 | 无 |
| `observation.images.cam_left_wrist` | `[3, 480, 640]` | AV1 | 50 | 无 |
| `observation.images.cam_right_wrist` | `[3, 480, 640]` | AV1 | 50 | 无 |

视频路径模板：

```text
videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
```

例如：

```text
videos/chunk-000/observation.images.cam_high/episode_000000.mp4
videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4
videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4
```

LeRobot 读取数据时，会根据 `episode_index`、`frame_index` 和 `video_path` 从对应 MP4 中解码图像帧。

## 语言任务和 episode 元数据

### `meta/tasks.jsonl`

`tasks.jsonl` 将 `task_index` 映射到自然语言指令：

```json
{"task_index": 0, "task": "Use the right arm to lift the bottle with white neck and red top head-up"}
{"task_index": 1, "task": "Lift the medium green bottle with capped neck off the table and hold it upright."}
{"task_index": 2, "task": "Pick the bottle with white neck and red top and keep it upright"}
```

### `meta/episodes.jsonl`

`episodes.jsonl` 每行对应一个 episode，包含：

- `episode_index`
- `tasks`
- `length`
- `action_config`

示例：

```json
{
  "episode_index": 0,
  "tasks": [
    "Use the right arm to lift the bottle with white neck and red top head-up"
  ],
  "length": 139,
  "action_config": [
    {
      "start_frame": 0,
      "end_frame": 139,
      "action_text": "Use the right arm to lift the bottle with white neck and red top head-up",
      "skill": ""
    }
  ]
}
```

需要注意：在 clean 单任务目录里，`total_tasks` 也是 50。也就是说，每个 episode 往往有自己的语言指令，而不是整个任务目录共用同一句 task。

### `meta/episodes_stats.jsonl`

`episodes_stats.jsonl` 保存每个 episode 的统计量，包括：

- `observation.state`
- `action`
- 三路视频图像
- `timestamp`
- `frame_index`
- `episode_index`
- `index`
- `task_index`

统计项包括：

```text
min
max
mean
std
count
```

这些统计量可用于归一化或数据质量检查。

## `latents/` 目录

`latents/` 是这个仓库额外提供的内容，不是 LeRobot 标准数据加载所必需。

README 中说明：

> Robotwin dataset in Lerobot format, with video latents already extracted in WAN 2.2 format, ready for use in Lingbot-VA post-training.

也就是说，这些 latent 是提前抽好的 WAN 2.2 视频 latent，主要用于 LingBot-VA post-training。

latent 文件路径示例：

```text
latents/chunk-000/observation.images.cam_high/episode_000000_0_139.pth
```

文件名包含：

- `episode_000000`：episode 编号
- `0_139`：帧范围

抽样检查一个 `.pth`，它是 PyTorch zip checkpoint，大致包含：

| 字段 | 含义 |
| --- | --- |
| `latent` | bfloat16 视频 latent tensor |
| `latent_num_frames` | latent 帧数，样例为 9 |
| `latent_height` | latent 高度，样例为 16 |
| `latent_width` | latent 宽度，样例为 20 |
| `video_num_frames` | 用于 latent 的视频帧数，样例为 33 |
| `video_height` | latent 输入视频高度，样例为 256 |
| `video_width` | latent 输入视频宽度，样例为 320 |
| `text_emb` | bfloat16 文本 embedding |
| `text` | 对应语言指令 |
| `frame_ids` | 抽帧 id，如 `0, 4, 8, ..., 128` |
| `start_frame` | 起始帧 |
| `end_frame` | 结束帧 |
| `fps` | latent 使用的 fps，样例为 12 |
| `ori_fps` | 原始视频 fps，样例为 50 |

样例中：

```text
latent shape: 约 (2880, 48)
text_emb shape: 约 (512, 4096)
video_num_frames: 33
video_height x video_width: 256 x 320
latent_num_frames x latent_height x latent_width: 9 x 16 x 20
fps: 12
ori_fps: 50
```

因此：

- 如果训练普通 LeRobot policy，主要使用 `data/`、`videos/`、`meta/`。
- 如果训练 LingBot-VA 或视频生成/视频表征相关模型，`latents/` 可以直接作为预提取特征使用。

## 训练使用视角

### 普通 LeRobot policy

需要关注：

```text
data/
videos/
meta/
```

典型读取逻辑：

1. 读取 `meta/info.json` 获得 schema、fps 和路径模板。
2. 读取 `data/chunk-000/episode_XXXXXX.parquet` 获得逐帧 state/action/index。
3. 通过 `video_path` 和视频 key 找到三路 MP4。
4. 用 `frame_index` 解码对应视频帧。
5. 用 `task_index` 到 `tasks.jsonl` 中取语言指令。

### VLA / video latent post-training

需要额外关注：

```text
latents/
```

latent 已经包含：

- 视频 latent
- 文本 embedding
- 原始语言文本
- 抽帧信息
- 原始 fps 与 latent fps 映射

这部分可以减少重复的视频编码成本。

## 数据形态总结

| 层级 | 文件类型 | 内容 |
| --- | --- | --- |
| 顶层目录 | folder | clean/aug 两套数据 |
| 任务目录 | folder | 一个 RoboTwin 任务的数据 |
| `data/*.parquet` | parquet | 逐帧低维 state/action |
| `videos/*.mp4` | MP4 | 三路相机视频 |
| `meta/info.json` | JSON | LeRobot schema、路径模板、fps、统计规模 |
| `meta/tasks.jsonl` | JSONL | `task_index` 到语言指令 |
| `meta/episodes.jsonl` | JSONL | episode 长度、任务文本、帧范围 |
| `meta/episodes_stats.jsonl` | JSONL | 每个 episode 的统计量 |
| `latents/*.pth` | PyTorch checkpoint | WAN 2.2 视频 latent 和文本 embedding |

## 关键判断

这个数据集的核心不是 Hugging Face Dataset Viewer 展示的 `video/label` 表，而是 LeRobot 原生目录结构。

最重要的三个判断是：

1. 它是 **LeRobot v2.1**，不是最新 v3.0。
2. 它是 **episode-per-file**：每个 episode 一个 parquet，每个 camera 一个 MP4。
3. 它的 action/state 是 **16 维 EEF pose + gripper**，不是普通关节角 action。

## 参考链接

- Hugging Face 数据集主页：https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot
- LeRobot GitHub：https://github.com/huggingface/lerobot
- LeRobot Dataset 文档：https://huggingface.co/docs/lerobot/lerobot-dataset-v3
- RoboTwin 相关文档：https://github.com/huggingface/lerobot/blob/main/docs/source/robotwin.mdx
