from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .deterministic import stable_seed


NEUTRAL = {
    "identity": 0.0, "brightness": 1.0, "gamma": 1.0,
    "color_temperature": 0.0, "gaussian_noise_std": 0.0,
    "gaussian_blur_sigma": 0.0,
}


def _float_clip(images) -> tuple[torch.Tensor, bool]:
    value = torch.as_tensor(np.asarray(images)).float()
    was_uint8 = np.asarray(images).dtype == np.uint8
    if was_uint8 or value.max() > 1.0:
        value = value / 255.0
    if value.ndim == 3:
        value = value[None]
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"Images must be [T,H,W,3], got {tuple(value.shape)}")
    return value.clamp(0, 1), was_uint8


def _restore(value: torch.Tensor, uint8: bool) -> np.ndarray:
    value = value.clamp(0, 1)
    if uint8:
        return (value * 255.0).round().byte().numpy()
    return value.numpy()


def gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return images
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coords / sigma).square())
    kernel = kernel / kernel.sum()
    value = images.permute(0, 3, 1, 2)
    channels = value.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    value = F.pad(value, (radius, radius, 0, 0), mode="reflect")
    value = F.conv2d(value, horizontal, groups=channels)
    value = F.pad(value, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(value, vertical, groups=channels).permute(0, 2, 3, 1)


def apply_perturbation(images, name: str, severity: float, seed: int) -> np.ndarray:
    value, uint8 = _float_clip(images)
    severity = float(severity)
    if name == "identity":
        result = value.clone()
    elif name == "brightness":
        result = value * severity
    elif name == "gamma":
        result = value.clamp_min(1e-6).pow(severity)
    elif name == "color_temperature":
        scale = value.new_tensor([1.0 + severity, 1.0, 1.0 - severity])
        result = value * scale
    elif name == "gaussian_noise_std":
        generator = torch.Generator(device="cpu").manual_seed(seed % (2**63 - 1))
        result = value + torch.randn(value.shape, generator=generator) * severity
    elif name == "gaussian_blur_sigma":
        result = gaussian_blur(value, severity)
    else:
        raise KeyError(f"Unknown nuisance perturbation: {name}")
    return _restore(result, uint8)


def perturb_cameras(
    cameras: dict[str, np.ndarray], name: str, severity: float,
    global_seed: int, sample_id: str, per_camera: bool = False,
) -> tuple[dict[str, np.ndarray], int]:
    shared_seed = stable_seed(global_seed, sample_id, f"augmentation:{name}:{severity}")
    result = {}
    for camera, images in cameras.items():
        seed = stable_seed(shared_seed, sample_id, camera) if per_camera else shared_seed
        result[camera] = apply_perturbation(images, name, severity, seed)
    return result, shared_seed


def direction(images, kind: str, seed: int, mask=None) -> torch.Tensor:
    value, _ = _float_clip(images)
    generator = torch.Generator(device="cpu").manual_seed(seed % (2**63 - 1))
    if kind == "photo":
        delta = value - value.mean(dim=(1, 2, 3), keepdim=True)
        delta = delta + torch.ones_like(value) * 0.25
    elif kind in {"background", "roi"}:
        if mask is None:
            raise ValueError(f"{kind} direction requires a mask")
        selected = torch.as_tensor(mask).bool()
        if selected.ndim == 2:
            selected = selected[None, ..., None]
        elif selected.ndim == 3:
            selected = selected[..., None]
        if kind == "background":
            selected = ~selected
        delta = torch.randn(value.shape, generator=generator) * selected
    elif kind == "geometry":
        delta = torch.roll(value, shifts=1, dims=2) - torch.roll(value, shifts=-1, dims=2)
    else:
        raise KeyError(f"Unknown direction kind: {kind}")
    return delta / delta.norm().clamp_min(1e-12)


def directional_pair(images, kind: str, epsilon: float, seed: int, mask=None):
    value, uint8 = _float_clip(images)
    delta = direction(value.numpy(), kind, seed, mask=mask)
    return _restore(value + epsilon * delta, uint8), _restore(value - epsilon * delta, uint8)


def configured_perturbations(config: dict[str, Any]):
    for name, severities in config.items():
        if name == "directional_epsilons":
            continue
        for severity in severities:
            yield name, float(severity)

