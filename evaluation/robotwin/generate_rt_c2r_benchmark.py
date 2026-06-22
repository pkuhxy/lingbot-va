"""Generate RT-C2R manifests and RoboTwin task_config YAML files."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.robotwin.rt_c2r import (  # noqa: E402
    SPLIT_RANDOMIZATION,
    TASK_NAMES,
    manifest_record,
    task_config_name,
)


def robotwin_root_from_env() -> Path:
    root = os.environ.get("ROBOTWIN_ROOT") or os.environ.get("ROBOWIN_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    return (PROJECT_ROOT / "RoboTwin").resolve()


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def write_manifests(
    output_dir: Path,
    episodes_per_task: int,
    base_seed: int,
    splits: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        out_file = output_dir / f"{task_config_name(split)}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for task_index, task_name in enumerate(TASK_NAMES):
                for episode_id in range(episodes_per_task):
                    record = manifest_record(
                        split=split,
                        task_name=task_name,
                        task_index=task_index,
                        episode_id=episode_id,
                        base_seed=base_seed,
                    )
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {out_file}")


def write_task_configs(
    robotwin_root: Path,
    template_name: str,
    splits: list[str],
    output_dir: Path | None,
) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for --write-task-configs. Install pyyaml in the "
            "RoboTwin evaluation environment, or run without --write-task-configs "
            "to generate only seed manifests."
        ) from exc

    task_config_dir = robotwin_root / "task_config"
    template_path = task_config_dir / f"{template_name}.yml"
    if not template_path.is_file():
        raise FileNotFoundError(
            f"Template task_config not found: {template_path}. "
            "Set ROBOTWIN_ROOT to a RoboTwin checkout that contains "
            "task_config/demo_clean.yml, or pass --template-name."
        )

    with template_path.open("r", encoding="utf-8") as f:
        template = yaml.safe_load(f)

    target_dir = output_dir or task_config_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        config = deep_update(
            template,
            {"domain_randomization": SPLIT_RANDOMIZATION[split]},
        )
        out_file = target_dir / f"{task_config_name(split)}.yml"
        with out_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        print(f"Wrote {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "robotwin" / "rt_c2r_manifests",
        help="Directory for rt_c2r_*.jsonl seed manifests.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=100,
        help="Number of eval seeds per task and split.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260622,
        help="Base seed used to derive deterministic per-episode seeds.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLIT_RANDOMIZATION.keys()),
        choices=list(SPLIT_RANDOMIZATION.keys()),
    )
    parser.add_argument(
        "--write-task-configs",
        action="store_true",
        help="Also write rt_c2r_*.yml task_config files.",
    )
    parser.add_argument(
        "--skip-manifests",
        action="store_true",
        help="Do not write seed manifests. Useful when only refreshing "
        "RoboTwin task_config YAML files.",
    )
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=robotwin_root_from_env(),
        help="RoboTwin checkout root used when writing task_config files.",
    )
    parser.add_argument(
        "--template-name",
        default="demo_clean",
        help="Base RoboTwin task_config name to patch for RT-C2R splits.",
    )
    parser.add_argument(
        "--task-config-output-dir",
        type=Path,
        default=None,
        help="Optional output directory for generated YAMLs. Defaults to "
        "<ROBOTWIN_ROOT>/task_config so the existing client can load them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_manifests:
        write_manifests(
            output_dir=args.manifest_dir,
            episodes_per_task=args.episodes_per_task,
            base_seed=args.base_seed,
            splits=args.splits,
        )
    if args.write_task_configs:
        write_task_configs(
            robotwin_root=args.robotwin_root.expanduser().resolve(),
            template_name=args.template_name,
            splits=args.splits,
            output_dir=args.task_config_output_dir,
        )


if __name__ == "__main__":
    main()
