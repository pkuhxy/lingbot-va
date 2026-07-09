"""Build an episode-disjoint, stable clean probe-window manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from common.io import write_jsonl


CAMERAS = (
    "observation.images.cam_high", "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def task_name(path: Path) -> str:
    for marker in ("-demo_", "-aloha-", "_clean_50"):
        if marker in path.name:
            return path.name.split(marker)[0]
    return path.name


def records(root: Path, episodes_per_task: int, windows_per_episode: int, seed: int, split: str):
    output: list[dict[str, Any]] = []
    task_dirs = sorted(file.parent.parent for file in root.glob("*/meta/info.json"))
    for task_dir in task_dirs:
        task = task_name(task_dir)
        episode_file = task_dir / "meta" / "episodes.jsonl"
        if not episode_file.is_file():
            raise FileNotFoundError(episode_file)
        episodes = [json.loads(line) for line in episode_file.read_text().splitlines() if line]
        rng = random.Random(f"{seed}:{task}")
        rng.shuffle(episodes)
        for episode in sorted(episodes[:episodes_per_task], key=lambda row: row["episode_index"]):
            episode_id = int(episode["episode_index"])
            windows = episode.get("action_config", [])
            if not windows:
                length = int(episode.get("length", 0))
                windows = [{"start_frame": 0, "end_frame": length}]
            rng.shuffle(windows)
            for action_config_index, window in enumerate(windows[:windows_per_episode]):
                start, end = int(window["start_frame"]), int(window["end_frame"])
                chunk = episode_id // 1000
                sample_id = f"clean/{task}/episode_{episode_id:06d}/{start}_{end}"
                output.append({
                    "sample_id": sample_id, "domain": "clean", "task": task,
                    "repo_id": str(task_dir.resolve()), "episode_index": episode_id,
                    "episode_or_seed": f"episode_{episode_id:06d}",
                    "start_frame": start, "end_frame": end,
                    "action_config_index": action_config_index, "split": split,
                    "prompt": (episode.get("tasks") or [task])[0],
                    "video_paths": {
                        camera: str(task_dir / "videos" / f"chunk-{chunk:03d}" / camera / f"episode_{episode_id:06d}.mp4")
                        for camera in CAMERAS
                    },
                })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dataset-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--split", choices=("clean_holdout", "clean_train_probe"), default="clean_train_probe")
    args = parser.parse_args()
    rows = records(
        args.clean_dataset_path.expanduser().resolve(), args.episodes_per_task,
        args.windows_per_episode, args.seed, args.split,
    )
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate stable sample IDs")
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} probe windows to {args.output}")


if __name__ == "__main__":
    main()

