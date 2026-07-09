from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open

from .io import sha256_file, sha256_paths


WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.pt", "*.pth")


def transformer_dir(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    candidate = path / "transformer"
    result = candidate if candidate.is_dir() else path
    if not (result / "config.json").is_file():
        raise FileNotFoundError(f"Transformer config not found: {result / 'config.json'}")
    return result


def weight_files(path: str | Path) -> list[Path]:
    root = transformer_dir(path)
    files = sorted({file for pattern in WEIGHT_GLOBS for file in root.glob(pattern)})
    index = root / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        with index.open("r", encoding="utf-8") as handle:
            names = set(json.load(handle)["weight_map"].values())
        files = [root / name for name in sorted(names)]
    if not files:
        raise FileNotFoundError(f"No transformer weights found under {root}")
    return files


def checkpoint_step(path: str | Path) -> int | None:
    for parent in [Path(path), *Path(path).parents]:
        match = re.fullmatch(r"checkpoint_step_(\d+)", parent.name)
        if match:
            return int(match.group(1))
    return None


def load_transformer_checkpoint(path: str | Path, dtype=torch.float32, device="cpu", attn_mode="flex"):
    from wan_va.modules.utils import load_transformer

    return load_transformer(
        str(transformer_dir(path)), torch_dtype=dtype,
        torch_device=device, attn_mode=attn_mode,
    )


class SafeTensorStore:
    """Name-indexed safetensors reader that never materializes a full checkpoint."""

    def __init__(self, path: str | Path):
        self.root = transformer_dir(path)
        self.files = weight_files(self.root)
        if any(file.suffix != ".safetensors" for file in self.files):
            raise ValueError("Streaming audit requires safetensors checkpoints")
        self.handles = {
            file: safe_open(str(file), framework="pt", device="cpu") for file in self.files
        }
        self.name_to_file: dict[str, Path] = {}
        for file, handle in self.handles.items():
            for name in handle.keys():
                if name in self.name_to_file:
                    raise ValueError(f"Duplicate parameter {name} in {self.root}")
                self.name_to_file[name] = file

    @property
    def names(self) -> set[str]:
        return set(self.name_to_file)

    def tensor(self, name: str) -> torch.Tensor:
        return self.handles[self.name_to_file[name]].get_tensor(name)


def module_group(name: str) -> str:
    for prefix in (
        "patch_embedding_mlp", "action_embedder", "condition_embedder_action",
        "condition_embedder", "norm_out", "proj_out", "action_proj_out",
    ):
        if name == prefix or name.startswith(prefix + "."):
            return prefix
    match = re.match(r"blocks\.(\d+)\.", name)
    return f"blocks.{match.group(1)}" if match else "other"


def _finite_count(tensor: torch.Tensor) -> int:
    if tensor.is_floating_point() or tensor.is_complex():
        return int((~torch.isfinite(tensor)).sum().item())
    return 0


def audit_pair(base: SafeTensorStore, other: SafeTensorStore, epsilon=1e-12) -> dict[str, Any]:
    missing = sorted(base.names - other.names)
    unexpected = sorted(other.names - base.names)
    shape_mismatch = []
    parameter_rows = []
    totals = defaultdict(lambda: {"delta_sq": 0.0, "base_sq": 0.0, "dot": 0.0, "nonfinite": 0, "numel": 0})
    for name in sorted(base.names & other.names):
        left, right = base.tensor(name), other.tensor(name)
        if left.shape != right.shape:
            shape_mismatch.append({"name": name, "base": list(left.shape), "other": list(right.shape)})
            continue
        left, right = left.float(), right.float()
        delta = right - left
        delta_sq = float(torch.dot(delta.flatten(), delta.flatten()).item())
        base_sq = float(torch.dot(left.flatten(), left.flatten()).item())
        dot = float(torch.dot(delta.flatten(), left.flatten()).item())
        delta_norm, base_norm = math.sqrt(delta_sq), math.sqrt(base_sq)
        relative = delta_norm / (base_norm + epsilon)
        cosine = dot / (delta_norm * base_norm + epsilon)
        nonfinite = _finite_count(left) + _finite_count(right)
        group = module_group(name)
        parameter_rows.append({
            "name": name, "group": group, "shape": list(left.shape),
            "relative_l2": relative, "update_cos": cosine,
            "nonfinite": nonfinite,
        })
        for key in (group, "__all__"):
            totals[key]["delta_sq"] += delta_sq
            totals[key]["base_sq"] += base_sq
            totals[key]["dot"] += dot
            totals[key]["nonfinite"] += nonfinite
            totals[key]["numel"] += left.numel()
    groups = {}
    for name, value in totals.items():
        delta_norm, base_norm = math.sqrt(value["delta_sq"]), math.sqrt(value["base_sq"])
        groups[name] = {
            "relative_l2": delta_norm / (base_norm + epsilon),
            "update_cos": value["dot"] / (delta_norm * base_norm + epsilon),
            "nonfinite": value["nonfinite"], "numel": value["numel"],
        }
    return {
        "missing_parameters": missing, "unexpected_parameters": unexpected,
        "shape_mismatch": shape_mismatch, "groups": groups,
        "parameters": parameter_rows,
    }


def align_head_audit(transformer_path: str | Path) -> dict[str, Any] | None:
    checkpoint = transformer_dir(transformer_path).parent
    head = checkpoint / "align_head"
    if not head.is_dir():
        return None
    files = sorted(head.glob("*"))
    result: dict[str, Any] = {
        "path": str(head), "files": {file.name: sha256_file(file) for file in files if file.is_file()}
    }
    weights = head / "model.safetensors"
    if weights.is_file():
        handle = safe_open(str(weights), framework="pt", device="cpu")
        result["parameter_norms"] = {
            name: float(handle.get_tensor(name).float().norm().item()) for name in handle.keys()
        }
    state_path = checkpoint / "trainer_state.pt"
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        result["trainer_step"] = state.get("step")
        result["lambda_align"] = state.get("lambda_align", state.get("config", {}).get("lambda_align") if isinstance(state.get("config"), dict) else None)
    return result


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    root = transformer_dir(path)
    weights = weight_files(root)
    return {
        "path": str(root), "step": checkpoint_step(root),
        "weights_sha256": sha256_paths(weights),
        "weight_files": {file.name: sha256_file(file) for file in weights},
        "config_sha256": sha256_file(root / "config.json"),
    }


def plot_parameter_drift(audit: dict[str, Any], output: str | Path) -> None:
    import matplotlib.pyplot as plt

    labels, series = [], defaultdict(list)
    comparisons = audit["comparisons"]
    all_groups = set()
    for value in comparisons.values():
        all_groups.update(value["groups"])
    labels = sorted(
        (group for group in all_groups if group != "__all__"),
        key=lambda x: (0, int(x.split(".")[1])) if x.startswith("blocks.") else (1, x),
    )
    for checkpoint, value in comparisons.items():
        series[checkpoint] = [value["groups"].get(group, {}).get("relative_l2", float("nan")) for group in labels]
    figure, axis = plt.subplots(figsize=(max(12, len(labels) * 0.35), 5))
    for checkpoint, values in series.items():
        axis.plot(range(len(labels)), values, marker="o", markersize=2, label=checkpoint)
    axis.set_xticks(range(len(labels)), labels, rotation=75, ha="right")
    axis.set_ylabel("relative L2")
    axis.set_yscale("symlog", linthresh=1e-8)
    axis.legend()
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

