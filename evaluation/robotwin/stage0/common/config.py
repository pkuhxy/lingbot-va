from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


PATH_KEYS = {
    "model_root", "clean_dataset_path", "clean_manifest", "paired_bank",
    "c2r_bank", "output_dir",
}


def _expand(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: _expand(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, key) for v in value]
    if isinstance(value, str):
        value = os.path.expandvars(os.path.expanduser(value))
        if key in PATH_KEYS or key in {"base", "clean", "contrastive", "aug"}:
            return str(Path(value))
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Stage-0 config must be a mapping: {path}")
    config = _expand(config)
    config["_config_path"] = str(path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = ("checkpoints", "model_root", "output_dir", "runtime", "train_forward")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    if "base" not in config["checkpoints"]:
        raise ValueError("checkpoints.base is required")
    if int(config["runtime"].get("batch_size", 1)) != 1:
        raise ValueError("Stage-0 first implementation requires runtime.batch_size=1")
    quantiles = config["train_forward"].get("timestep_quantiles", [])
    if any(not 0.0 <= float(q) <= 1.0 for q in quantiles):
        raise ValueError("timestep quantiles must lie in [0, 1]")


def checkpoint_items(config: dict[str, Any]) -> list[tuple[str, Path]]:
    return [(name, Path(path).expanduser().resolve()) for name, path in config["checkpoints"].items()]


def with_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.update({key: value for key, value in overrides.items() if value is not None})
    validate_config(result)
    return result


def torch_dtype(name: str):
    import torch

    aliases = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc

