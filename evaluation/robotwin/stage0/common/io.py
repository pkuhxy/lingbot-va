from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


METRIC_COLUMNS = [
    "run_id", "checkpoint", "sample_id", "task", "episode_or_seed",
    "domain", "c2r_split", "progress_bin", "layer", "token_group",
    "perturbation", "severity", "timestep_quantile", "timestep", "sigma",
    "noise_seed", "transform_seed", "metric_name", "metric_value",
]


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames=None) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = set().union(*(row.keys() for row in rows)) if rows else set()
        fieldnames = METRIC_COLUMNS + sorted(keys.difference(METRIC_COLUMNS))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_paths(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(str(path.name).encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_record() -> dict[str, Any]:
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        torch_version = cuda_version = gpu = None
    try:
        import diffusers
        diffusers_version = diffusers.__version__
    except ImportError:
        diffusers_version = None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch_version, "cuda": cuda_version,
        "diffusers": diffusers_version, "gpu": gpu,
    }


def create_run_dir(config: dict[str, Any], run_id: str | None = None) -> Path:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config["output_dir"]).expanduser().resolve() / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run already exists; refusing to overwrite: {run_dir}")
    for name in (
        "manifests", "checkpoint_audit", "features", "paired_predictions",
        "metrics", "plots",
    ):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    resolved = {k: v for k, v in config.items() if not k.startswith("_")}
    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    write_json(run_dir / "environment.json", environment_record())
    return run_dir


def get_run_dir(config: dict[str, Any], run_dir: str | Path | None, run_id=None) -> Path:
    if run_dir is not None:
        path = Path(run_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {path}")
        return path
    return create_run_dir(config, run_id)


def metric_row(**values: Any) -> dict[str, Any]:
    row = {key: "" for key in METRIC_COLUMNS}
    row.update(values)
    value = row.get("metric_value")
    if value is not None and value != "":
        numeric = float(value)
        if not (numeric == numeric and abs(numeric) != float("inf")):
            raise ValueError(f"Non-finite metric: {row}")
        row["metric_value"] = numeric
    return row


def sample_storage_name(sample_id: str) -> str:
    readable = sample_id.replace("/", "__").replace(" ", "_")
    suffix = hashlib.sha256(sample_id.encode()).hexdigest()[:12]
    return f"{readable[:120]}__{suffix}"

