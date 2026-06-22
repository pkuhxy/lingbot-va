"""Summarize RoboTwin RT-C2R benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

SPLITS = ["easy", "background", "light", "clutter", "height", "hard"]


def load_task_metric(task_dir: Path) -> dict[str, float] | None:
    res_file = task_dir / "res.json"
    if res_file.is_file():
        with res_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "success": float(data.get("succ_num", 0.0)),
            "total": float(data.get("total_num", 0.0)),
        }

    episode_file = task_dir / "episodes.jsonl"
    if episode_file.is_file():
        success = 0.0
        total = 0.0
        with episode_file.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                success += 1.0 if record.get("success") else 0.0
                total += 1.0
        return {"success": success, "total": total}

    return None


def collect_split_metrics(split_root: Path) -> dict[str, dict[str, float]]:
    metrics_root = split_root / "metrics"
    candidates: list[Path] = []
    if metrics_root.is_dir():
        candidates.append(metrics_root)
    candidates.extend(sorted(split_root.glob("stseed-*/metrics")))

    task_metrics: dict[str, dict[str, float]] = {}
    for root in candidates:
        for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            metric = load_task_metric(task_dir)
            if metric is None:
                continue
            task = task_dir.name
            agg = task_metrics.setdefault(task, {"success": 0.0, "total": 0.0})
            agg["success"] += metric["success"]
            agg["total"] += metric["total"]

    for metric in task_metrics.values():
        total = metric["total"]
        metric["success_rate"] = metric["success"] / total if total > 0 else None
    return task_metrics


def summarize(root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"splits": {}, "per_task": {}}
    for split in SPLITS:
        split_metrics = collect_split_metrics(root / split)
        total_success = sum(v["success"] for v in split_metrics.values())
        total = sum(v["total"] for v in split_metrics.values())
        split_rate = total_success / total if total > 0 else None
        summary["splits"][split] = {
            "success": total_success,
            "total": total,
            "success_rate": split_rate,
            "num_tasks": len(split_metrics),
        }
        for task, metric in split_metrics.items():
            summary["per_task"].setdefault(task, {})[split] = metric

    easy = summary["splits"]["easy"]["success_rate"]
    hard = summary["splits"]["hard"]["success_rate"]
    summary["retention_hard_over_easy"] = (
        hard / easy if easy not in (None, 0) and hard is not None else None
    )
    summary["drop_easy_minus_hard"] = (
        easy - hard if easy is not None and hard is not None else None
    )
    return summary


def write_csv(summary: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "name", "split", "success", "total", "success_rate"])
        for split, metric in summary["splits"].items():
            writer.writerow(
                [
                    "split",
                    "all",
                    split,
                    metric["success"],
                    metric["total"],
                    metric["success_rate"],
                ]
            )
        for task, split_metrics in sorted(summary["per_task"].items()):
            for split, metric in sorted(split_metrics.items()):
                writer.writerow(
                    [
                        "task",
                        task,
                        split,
                        metric["success"],
                        metric["total"],
                        metric["success_rate"],
                    ]
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="RT-C2R result root.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(args.root.expanduser().resolve())
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.csv_output is not None:
        write_csv(summary, args.csv_output)


if __name__ == "__main__":
    main()

