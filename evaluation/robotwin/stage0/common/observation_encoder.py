from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


class RobotwinObservationEncoder:
    """Server-equivalent, resettable three-camera T-shape VAE encoder."""

    def __init__(self, model_root, device="cuda:0", dtype=torch.bfloat16):
        from wan_va.modules.utils import WanVAEStreamingWrapper, load_vae

        self.device, self.dtype = torch.device(device), dtype
        vae_path = Path(model_root).expanduser().resolve() / "vae"
        self.vae = load_vae(str(vae_path), torch_dtype=dtype, torch_device=self.device)
        half_vae = load_vae(str(vae_path), torch_dtype=dtype, torch_device=self.device)
        self.streaming = WanVAEStreamingWrapper(self.vae)
        self.streaming_half = WanVAEStreamingWrapper(half_vae)

    def clear_cache(self) -> None:
        self.streaming.clear_cache()
        self.streaming_half.clear_cache()

    @staticmethod
    def _camera_tensor(images, height: int, width: int) -> torch.Tensor:
        value = torch.as_tensor(np.asarray(images)).float()
        if value.ndim == 3:
            value = value[None]
        if value.ndim != 4 or value.shape[-1] != 3:
            raise ValueError(f"Camera clip must be [T,H,W,3], got {tuple(value.shape)}")
        value = value.permute(3, 0, 1, 2)
        value = F.interpolate(value, size=(height, width), mode="bilinear", align_corners=False)
        return value[None]

    @torch.no_grad()
    def encode(self, cameras: Mapping[str, np.ndarray], reset: bool = True) -> torch.Tensor:
        if reset:
            self.clear_cache()
        missing = set(CAMERA_KEYS).difference(cameras)
        if missing:
            raise KeyError(f"Missing RoboTwin cameras: {sorted(missing)}")
        high = self._camera_tensor(cameras[CAMERA_KEYS[0]], 256, 320)
        wrists = torch.cat([
            self._camera_tensor(cameras[key], 128, 160) for key in CAMERA_KEYS[1:]
        ], dim=0)
        high = (high / 255.0 * 2.0 - 1.0).to(self.device, self.dtype)
        wrists = (wrists / 255.0 * 2.0 - 1.0).to(self.device, self.dtype)
        encoded_high = self.streaming.encode_chunk(high)
        encoded_wrists = self.streaming_half.encode_chunk(wrists)
        encoded = torch.cat([
            torch.cat(encoded_wrists.split(1, dim=0), dim=-1), encoded_high
        ], dim=-2)
        mu, _logvar = torch.chunk(encoded, 2, dim=1)
        mean = torch.tensor(self.vae.config.latents_mean, device=mu.device).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=mu.device).view(1, -1, 1, 1, 1)
        normalized = ((mu.float() - mean) / std).to(mu.dtype)
        return torch.cat(normalized.split(1, dim=0), dim=-1).to(self.device)

