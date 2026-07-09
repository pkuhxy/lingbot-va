"""Visualize RT-C2R candidate cases rejected by RoboTwin's scripted expert.

This script reads validation failure logs produced by
generate_rt_c2r_validated_manifests.py, selects 1-2 rejected candidate seeds per
split, and optionally replays the RoboTwin scripted expert to save camera
snapshots.  It does not require a LingBot-VA inference server because it only
uses RoboTwin's built-in expert/planner.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.robotwin.generate_rt_c2r_benchmark import write_task_configs  # noqa: E402
from evaluation.robotwin.generate_rt_c2r_validated_manifests import (  # noqa: E402
    build_task_args,
    class_decorator,
    close_env_quietly,
    configure_robotwin_imports,
    ensure_egl_vendor_dirs,
    ensure_required_assets,
    robotwin_root_from_env,
)
from evaluation.robotwin.rt_c2r import SPLIT_RANDOMIZATION, TASK_NAMES  # noqa: E402


DEFAULT_SPLITS = ["easy", "background", "light", "clutter", "height", "hard"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_infrastructure_failure(record: dict[str, Any]) -> bool:
    reason = str(record.get("failure_reason", ""))
    infrastructure_patterns = [
        "No such file or directory",
        "Missing RoboTwin assets",
        "Task config not found",
        "background_texture",
        "egl_vendor",
    ]
    return any(pattern in reason for pattern in infrastructure_patterns)


def reason_bucket(record: dict[str, Any]) -> str:
    reason = str(record.get("failure_reason", "unknown"))
    if reason.startswith("AssertionError"):
        return "AssertionError"
    if reason.startswith("FileNotFoundError"):
        return "FileNotFoundError"
    if reason.startswith("RuntimeError"):
        return "RuntimeError"
    if reason in ("unstable", "expert_plan_or_success_check_failed"):
        return reason
    return reason.split(":", 1)[0]


def collect_failure_records(
    failure_log_dir: Path,
    splits: list[str],
    tasks: set[str] | None,
    cases_per_split: int,
    include_infrastructure: bool,
    reason_contains: str | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for split in splits:
        split_records: list[dict[str, Any]] = []
        for failure_file in sorted((failure_log_dir / split).glob("*.jsonl")):
            for record in read_jsonl(failure_file):
                if record.get("split") != split:
                    continue
                if tasks is not None and record.get("task") not in tasks:
                    continue
                if not include_infrastructure and is_infrastructure_failure(record):
                    continue
                if reason_contains and reason_contains not in str(record.get("failure_reason", "")):
                    continue
                split_records.append(record)

        split_records.sort(
            key=lambda item: (
                str(item.get("task", "")),
                int(item.get("candidate_id", item.get("episode_id", 0))),
                int(item.get("seed", 0)),
            )
        )

        diverse: list[dict[str, Any]] = []
        seen_buckets: set[str] = set()
        for record in split_records:
            bucket = reason_bucket(record)
            if bucket in seen_buckets:
                continue
            diverse.append(record)
            seen_buckets.add(bucket)
            if len(diverse) >= cases_per_split:
                break
        if len(diverse) < cases_per_split:
            used = {id(record) for record in diverse}
            for record in split_records:
                if id(record) in used:
                    continue
                diverse.append(record)
                if len(diverse) >= cases_per_split:
                    break
        selected.extend(diverse[:cases_per_split])
    return selected


def normalize_image(image: Any, height: int | None = None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] > 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        if arr.max(initial=0) <= 1.0001:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if height is not None and arr.shape[0] != height:
        width = max(1, int(arr.shape[1] * height / arr.shape[0]))
        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(arr)


def title_bar(image: np.ndarray, title: str) -> np.ndarray:
    bar_height = 34
    bar = np.zeros((bar_height, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        bar,
        title[:80],
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, image])


def observation_grid(observation: dict[str, Any], title: str) -> np.ndarray | None:
    try:
        cameras = observation["observation"]
        images = [
            ("head", cameras["head_camera"]["rgb"]),
            ("left wrist", cameras["left_camera"]["rgb"]),
            ("right wrist", cameras["right_camera"]["rgb"]),
        ]
    except Exception:
        return None

    normalized = []
    base_height = normalize_image(images[0][1]).shape[0]
    for camera_name, image in images:
        normalized.append(title_bar(normalize_image(image, height=base_height), camera_name))
    grid = np.hstack(normalized)
    return title_bar(grid, title)


def save_observation_grid(observation: dict[str, Any] | None, path: Path, title: str) -> str | None:
    if observation is None:
        return None
    grid = observation_grid(observation, title)
    if grid is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return str(path)


def safe_get_obs(task_env) -> dict[str, Any] | None:
    try:
        return task_env.get_obs()
    except Exception:
        return None


def safe_check_success(task_env) -> bool | None:
    try:
        return bool(task_env.check_success())
    except Exception:
        return None


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "case"


def replay_expert_failure(
    record: dict[str, Any],
    robotwin_root: Path,
    output_dir: Path,
    case_index: int,
) -> dict[str, Any]:
    split = str(record["split"])
    task_name = str(record["task"])
    seed = int(record["seed"])
    candidate_id = int(record.get("candidate_id", record.get("episode_id", case_index)))
    episode_id = int(record.get("episode_id", 0))

    case_dir = (
        output_dir
        / "cases"
        / split
        / f"{case_index:02d}_{slugify(task_name)}_candidate{candidate_id}_seed{seed}"
    )
    case_dir.mkdir(parents=True, exist_ok=True)

    replay: dict[str, Any] = {
        "case_index": case_index,
        "split": split,
        "task": task_name,
        "seed": seed,
        "candidate_id": candidate_id,
        "episode_id": episode_id,
        "original_failure_reason": record.get("failure_reason"),
        "randomization": SPLIT_RANDOMIZATION[split],
        "case_dir": str(case_dir),
    }

    task_env = None
    initial_obs = None
    final_obs = None
    try:
        args = build_task_args(robotwin_root, task_name, split)
        args["render_freq"] = 0
        args["eval_video_log"] = False
        task_env = class_decorator(task_name)
        task_env.suc = 0
        task_env.test_num = 0

        task_env.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **args)
        initial_obs = safe_get_obs(task_env)
        replay["setup_success"] = True

        try:
            task_env.play_once()
            replay["play_once_exception"] = None
        except Exception as exc:
            replay["play_once_exception"] = f"{type(exc).__name__}: {exc}"

        final_obs = safe_get_obs(task_env)
        replay["plan_success"] = bool(getattr(task_env, "plan_success", False))
        replay["check_success"] = safe_check_success(task_env)
    except Exception as exc:
        replay["setup_or_replay_exception"] = f"{type(exc).__name__}: {exc}"
    finally:
        if task_env is not None:
            close_env_quietly(task_env)

    replay["initial_image"] = save_observation_grid(
        initial_obs,
        case_dir / "initial.png",
        f"initial | {split} | {task_name} | seed={seed}",
    )
    replay["after_expert_image"] = save_observation_grid(
        final_obs,
        case_dir / "after_expert.png",
        f"after expert | plan={replay.get('plan_success')} | check={replay.get('check_success')}",
    )
    write_json(case_dir / "case.json", replay)
    return replay


def relative(path: str | None, base: Path) -> str | None:
    if not path:
        return None
    path_obj = Path(path)
    try:
        return path_obj.relative_to(base).as_posix()
    except ValueError:
        return path_obj.as_posix()


def write_html_report(output_dir: Path, replays: list[dict[str, Any]]) -> Path:
    report_path = output_dir / "rt_c2r_expert_failure_gallery.html"
    sections: list[str] = []
    for replay in replays:
        initial = relative(replay.get("initial_image"), report_path.parent)
        after = relative(replay.get("after_expert_image"), report_path.parent)
        image_html = ""
        if initial:
            image_html += f'<figure><img src="{html.escape(initial)}"><figcaption>Initial scene</figcaption></figure>'
        if after:
            image_html += f'<figure><img src="{html.escape(after)}"><figcaption>After scripted expert attempt</figcaption></figure>'
        if not image_html:
            image_html = "<p class='muted'>没有保存到相机图，通常说明 setup_demo 在渲染前已经失败。</p>"

        reason = html.escape(str(replay.get("original_failure_reason")))
        exception = html.escape(str(replay.get("play_once_exception") or replay.get("setup_or_replay_exception") or ""))
        randomization = html.escape(json.dumps(replay.get("randomization"), ensure_ascii=False))
        sections.append(
            f"""
            <section class="case">
              <h2>{html.escape(str(replay["split"]))} / {html.escape(str(replay["task"]))}</h2>
              <p><b>seed:</b> {int(replay["seed"])}
                 <b>candidate_id:</b> {int(replay["candidate_id"])}
                 <b>episode_id:</b> {int(replay["episode_id"])}</p>
              <p><b>原始筛除原因:</b> <code>{reason}</code></p>
              <p><b>重放异常:</b> <code>{exception}</code></p>
              <p><b>plan_success:</b> {html.escape(str(replay.get("plan_success")))}
                 <b>check_success:</b> {html.escape(str(replay.get("check_success")))}</p>
              <p><b>随机化配置:</b> <code>{randomization}</code></p>
              <div class="images">{image_html}</div>
            </section>
            """
        )

    body = "\n".join(sections) if sections else "<p>没有选中任何 expert failure case。</p>"
    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>RT-C2R Expert Failure Gallery</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; }}
    .case {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 18px 0; }}
    .muted {{ color: #666; }}
    .images {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    figure {{ margin: 0; }}
    img {{ max-width: 720px; border: 1px solid #ddd; }}
    figcaption {{ color: #555; margin-top: 6px; }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>RT-C2R 被 RoboTwin Scripted Expert 筛除的候选样例</h1>
  <p class="muted">这些 case 不是 policy 失败，而是在 benchmark 构建阶段因为 scripted expert/CuRobo/check_success 失败而被排除。</p>
  {body}
</body>
</html>
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failure-log-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "robotwin" / "rt_c2r_validation_failures",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "rt_c2r_expert_failure_gallery",
    )
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS, choices=DEFAULT_SPLITS)
    parser.add_argument("--tasks", nargs="+", default=None, choices=TASK_NAMES)
    parser.add_argument("--cases-per-split", type=int, default=2)
    parser.add_argument("--reason-contains", default=None)
    parser.add_argument("--include-infrastructure", action="store_true")
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Only write selected_expert_failures.jsonl; do not replay RoboTwin.",
    )
    parser.add_argument("--robotwin-root", type=Path, default=robotwin_root_from_env())
    parser.add_argument("--gpu-id", default=None, help="Optional CUDA_VISIBLE_DEVICES value for replay.")
    parser.add_argument("--write-task-configs", action="store_true")
    parser.add_argument("--template-name", default="demo_clean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failure_log_dir = args.failure_log_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = collect_failure_records(
        failure_log_dir=failure_log_dir,
        splits=args.splits,
        tasks=None if args.tasks is None else set(args.tasks),
        cases_per_split=args.cases_per_split,
        include_infrastructure=args.include_infrastructure,
        reason_contains=args.reason_contains,
    )
    write_jsonl(output_dir / "selected_expert_failures.jsonl", selected)

    if args.select_only:
        print(f"Selected {len(selected)} expert failure records.")
        print(f"Selected failures: {output_dir / 'selected_expert_failures.jsonl'}")
        return

    robotwin_root = args.robotwin_root.expanduser().resolve()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    ensure_egl_vendor_dirs()
    ensure_required_assets(robotwin_root, args.splits)
    if args.write_task_configs:
        write_task_configs(
            robotwin_root=robotwin_root,
            template_name=args.template_name,
            splits=args.splits,
            output_dir=None,
        )
    configure_robotwin_imports(robotwin_root)

    replays = [
        replay_expert_failure(record, robotwin_root, output_dir, case_index=i)
        for i, record in enumerate(selected)
    ]
    write_jsonl(output_dir / "replayed_expert_failures.jsonl", replays)
    report_path = write_html_report(output_dir, replays)

    print(f"Selected {len(selected)} expert failure records.")
    print(f"Replay records: {output_dir / 'replayed_expert_failures.jsonl'}")
    print(f"HTML report: {report_path}")


if __name__ == "__main__":
    main()
