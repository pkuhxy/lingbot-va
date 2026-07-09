"""Clean manifest entry point with RoboTwin task-name normalization."""

from pathlib import Path

import build_clean_probe_manifest as implementation


def normalized_task_name(path: Path) -> str:
    name = path.name
    for marker in ("-demo_", "-aloha-", "-piper_", "_clean_50"):
        if marker in name:
            return name.split(marker)[0]
    return name


implementation.task_name = normalized_task_name


if __name__ == "__main__":
    implementation.main()

