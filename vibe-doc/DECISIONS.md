# 决策

## 决策记录

## 2026-06-05 - 使用中文维护 Vibe Docs

- 状态: Accepted
- 背景: 用户用中文要求整理并解释 `wan_va/` 代码。
- 决策: `vibe-doc/` 使用中文维护，代码名、配置名、命令保留原文。
- 影响: 后续 agent 应继续用中文更新这些文档，除非用户另有要求。
- 备选方案: 使用英文以贴近上游 README；未采用，因为当前用户请求是中文。

## 2026-06-05 - 为 `wan_va/` 增加 `ARCHITECTURE.md`

- 状态: Accepted
- 背景: 标准 `vibe-doc` 文件偏项目状态与交接；用户明确要求重点整理 `wan_va/` 代码并解释关键内容。
- 决策: 增加可选文件 `vibe-doc/ARCHITECTURE.md`，作为 `wan_va/` 代码讲解的主文档。
- 影响: `PROJECT.md` 保持项目级稳定概览；具体代码结构、流程和风险集中放在 `ARCHITECTURE.md`。
- 备选方案: 把所有说明放进 `PROJECT.md`；未采用，因为会让项目概览过长。

## 2026-06-05 - 只记录风险，不修改运行时代码

- 状态: Accepted
- 背景: 工作区已有本地代码改动，用户请求是文档整理，不是修 bug。
- 决策: 不改运行时代码；把观察到的风险和开放问题记录到 `STATUS.md`、`TASKS.md`、`AI_HANDOFF.md`。
- 影响: 不会回退或覆盖用户改动；后续代码修复作为明确的任务处理。
- 备选方案: 直接修复 config、dataset、packaging 风险；未采用，因为超出本次文档整理范围。

## 2026-06-15 - Stage-1 Latent 后训练只训练 Probe

- 状态: Accepted
- 背景: 新增 `wan_va/train_latent_posttrain.py` 用于 WAM 内部 video/action 表征对齐研究，目标是先学习 controllable dynamics subspace，而不是直接改动主模型。
- 决策: Stage-1 加载并冻结 pretrained transformer，只训练 `wan_va/modules/latent_posttraining.py` 中的 `WAMControllableSubspaceProbe`。默认从 input embedding 抽取 token，也可通过 `--hidden-source transformer` 使用 transformer hidden。
- 影响: checkpoint 只保存 probe/optimizer/config，不是可直接替换 transformer 的完整策略模型；后续 Stage-2 需要显式加载 probe 或把 latent consistency loss 接入 action path 后训练。
- 备选方案: Stage-1 直接微调 transformer；未采用，因为会混合“表征探针是否有效”和“主模型参数更新是否有效”两个变量。

## 2026-06-15 - RoboTwin Latent 后训练使用预提取 WAN Latents

- 状态: Accepted
- 背景: `robbyant/robotwin-clean-and-aug-lerobot` 是 LeRobot v2.1 episode-per-file 仓库，同时额外提供 `latents/*.pth`，其中包含 WAN 2.2 视频 latent、text embedding 和抽帧信息。
- 决策: `train_latent_posttrain.py` 的数据路径应指向包含 `empty_emb.pt`、clean/aug 子目录、任务目录、`meta/` 和 `latents/` 的数据集顶层。训练时读取 parquet action 与 `.pth` latent/text embedding，不直接解码 MP4。
- 影响: 启动脚本会检查 `empty_emb.pt`、`meta/info.json` 和 `latents/`；如果传入 clean/aug 子目录或单任务目录，需要额外处理 `empty_emb.pt` 路径。
- 备选方案: 训练时从 `videos/*.mp4` 重新编码 VAE latent；未采用，因为数据集已提供 WAN 2.2 latent，可节省大量视频编码成本并降低运行复杂度。

## 2026-06-15 - 保持 30 维标准 Action 布局

- 状态: Accepted
- 背景: RoboTwin parquet 中的 `action` 是 16 维 EEF pose + gripper，但 LingBot-VA 模型和推理端使用标准 30 维 action 布局。
- 决策: 不把 `action_dim` 改成 16。Dataset 把 16 维有效动作映射到 30 维布局中的有效 channel，并用 `actions_mask` 屏蔽无效 channel。
- 影响: Stage-1 probe、transformer action embedding 和服务端推理保持同一 action contract；新增数据集时需要维护 `used_action_channel_ids`、`inverse_used_action_channel_ids` 和归一化统计。
- 备选方案: 为 RoboTwin 训练单独 16 维 action head；未采用，因为会破坏现有模型/服务端 action contract。

## 2026-06-24 - RT-C2R 结果按 Checkpoint Case 隔离

- 状态: Accepted
- 背景: 用户要依次评测四个 checkpoint，并要求所有结果保存到 `c2r_bench_result`，每个 case 对应一个子文件夹。
- 决策: 为四个 checkpoint 创建 `evaluation/robotwin/run_rt_c2r_*.sh` wrapper，默认 `TEST_NUM=3`，使用 validated manifests，并把结果、client logs、server logs、pid 和 summary 写入 `c2r_bench_result/<case>/`。
- 影响: 后续运行不需要手写长命令；重复跑同一 case 会进入同一个 case 目录，必要时应人工清理旧结果或换 `RESULT_ROOT`。
- 备选方案: 单个脚本循环四个 checkpoint；未采用，因为用户明确要求四个 bash 文件，并且 server 切 checkpoint 需要逐个 case 控制。

## 2026-06-24 - Server 支持训练 Checkpoint 目录只替换 Transformer

- 状态: Accepted
- 背景: `checkpoint_step_5000` 训练输出只包含 `transformer/` 和训练附属文件，不包含完整模型的 `vae/ tokenizer/ text_encoder/`。直接作为 `--pretrained-model` 会触发 `checkpoint_step_5000/vae/config.json` 缺失错误。
- 决策: `wan_va/wan_va_server.py` 识别两类 `--pretrained-model`：完整模型目录按原样加载；训练 checkpoint 目录则保留 config 中的 base model root，只把 `transformer_model_name_or_path` 指向 checkpoint 的 `transformer/`。
- 影响: 本地 `ckpts/.../checkpoint_step_5000` 可直接用于 server 启动；完整发布模型如 `lingbot-va-base` 和 `lingbot-va-posttrain-robotwin` 仍按原逻辑加载。
- 备选方案: 手动把 `vae/ tokenizer/ text_encoder/` 复制或软链接到每个训练 checkpoint；未采用，因为会制造重复大文件并增加 checkpoint 管理成本。
