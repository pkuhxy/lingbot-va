from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .observation_encoder import CAMERA_KEYS


def ensure_batch(tensor: torch.Tensor, expected_ndim: int) -> torch.Tensor:
    return tensor[None] if tensor.ndim == expected_ndim - 1 else tensor


def load_tensor_sample(record: dict[str, Any], variant: str = "clean", device="cpu") -> dict[str, Any]:
    if "tensor_path" not in record:
        raise KeyError(f"Bank record {record.get('sample_id')} has no tensor_path")
    payload = torch.load(record["tensor_path"], map_location="cpu", weights_only=False)
    if variant == "clean":
        latents = payload["latents"]
    else:
        try:
            latents = payload["variants"][variant]["latents"]
        except KeyError as exc:
            raise KeyError(f"Variant {variant!r} missing for {record['sample_id']}") from exc
    sample = {
        "latents": ensure_batch(latents, 5),
        "actions": ensure_batch(payload["actions"], 5),
        "actions_mask": ensure_batch(payload["actions_mask"], 5),
        "text_emb": ensure_batch(payload["text_emb"], 3),
    }
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in sample.items()}


def load_rgb_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in CAMERA_KEYS}


def save_rgb_npz(path: str | Path, cameras: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(cameras[key]) for key in CAMERA_KEYS})


def variant_name(perturbation: str, severity: float) -> str:
    return f"{perturbation}__{severity:+.6g}".replace("+", "p").replace("-", "m").replace(".", "d")


def action_velocity(output, frames: int) -> torch.Tensor:
    action = output[1] if isinstance(output, (tuple, list)) else output
    batch, length, channels = action.shape
    if length % frames:
        raise ValueError(f"Action token length {length} is not divisible by frames={frames}")
    return action.reshape(batch, frames, length // frames, channels).permute(0, 3, 1, 2).unsqueeze(-1)

