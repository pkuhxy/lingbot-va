"""Build a clean-only RoboTwin training manifest from LeRobot task folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def infer_task_name(task_dir: Path) -> str:
    name = task_dir.name
    for marker in ("-demo_", "-aloha-", "_clean_50"):
        if marker in name:
            return name.split(marker)[0]
    return name


def iter_episode_records(
    clean_dataset_root: Path,
    max_episodes_per_task: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    task_dirs = sorted(
        p.parent.parent
        for p in clean_dataset_root.glob("*/meta/info.json")
        if p.is_file()
    )
    for task_index, task_dir in enumerate(task_dirs):
        task_name = infer_task_name(task_dir)
        episodes_file = task_dir / "meta" / "episodes.jsonl"
        if not episodes_file.is_file():
            raise FileNotFoundError(f"Missing episodes.jsonl: {episodes_file}")
        with episodes_file.open("r", encoding="utf-8") as f:
            for local_count, line in enumerate(f):
                if local_count >= max_episodes_per_task:
                    break
                episode = json.loads(line)
                episode_index = int(episode["episode_index"])
                records.append(
                    {
                        "benchmark": "robotwin_rt_c2r",
                        "split": "train_clean",
                        "task": task_name,
                        "task_index": task_index,
                        "episode_id": episode_index,
                        "demonstration_path": str(task_dir),
                        "data_path": str(
                            task_dir
                            / "data"
                            / f"chunk-{episode_index // 1000:03d}"
                            / f"episode_{episode_index:06d}.parquet"
                        ),
                        "instruction": episode.get("tasks", [None])[0],
                        "embodiment": "aloha-agilex",
                        "camera": [
                            "observation.images.cam_high",
                            "observation.images.cam_left_wrist",
                            "observation.images.cam_right_wrist",
                        ],
                        "randomization": {
                            "random_background": False,
                            "random_light": False,
                            "cluttered_table": False,
                            "random_table_height": 0,
                            "random_head_camera_dis": 0,
                            "random_embodiment": False,
                        },
                    }
                )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-dataset-root",
        type=Path,
        required=True,
        help="Path to lerobot_robotwin_eef_clean_50.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("train_manifests")
        / "robotwin_clean_50tasks_50demos_per_task.jsonl",
    )
    parser.add_argument("--max-episodes-per-task", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = iter_episode_records(
        clean_dataset_root=args.clean_dataset_root.expanduser().resolve(),
        max_episodes_per_task=args.max_episodes_per_task,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

