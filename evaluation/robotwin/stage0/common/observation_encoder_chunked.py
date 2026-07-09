"""Training-aligned chunked RoboTwin observation encoder."""

from __future__ import annotations

import torch

from .observation_encoder import CAMERA_KEYS, RobotwinObservationEncoder


class ChunkedRobotwinObservationEncoder(RobotwinObservationEncoder):
    """Encode a 4n+1 RGB clip as one initial frame followed by 4-frame chunks."""

    @staticmethod
    def _encode_stream(wrapper, video: torch.Tensor) -> torch.Tensor:
        frames = video.shape[2]
        if frames < 1 or (frames - 1) % 4:
            raise ValueError(
                f"Wan VAE streaming input must contain 4n+1 frames, got {frames}"
            )
        outputs = [wrapper.encode_chunk(video[:, :, :1])]
        outputs.extend(
            wrapper.encode_chunk(video[:, :, start:start + 4])
            for start in range(1, frames, 4)
        )
        return torch.cat(outputs, dim=2)

    @torch.no_grad()
    def encode(self, cameras, reset: bool = True) -> torch.Tensor:
        if reset:
            self.clear_cache()
        missing = set(CAMERA_KEYS).difference(cameras)
        if missing:
            raise KeyError(f"Missing RoboTwin cameras: {sorted(missing)}")
        high = self._camera_tensor(cameras[CAMERA_KEYS[0]], 256, 320)
        wrists = torch.cat(
            [self._camera_tensor(cameras[key], 128, 160) for key in CAMERA_KEYS[1:]],
            dim=0,
        )
        high = (high / 255.0 * 2.0 - 1.0).to(self.device, self.dtype)
        wrists = (wrists / 255.0 * 2.0 - 1.0).to(self.device, self.dtype)
        encoded_high = self._encode_stream(self.streaming, high)
        encoded_wrists = self._encode_stream(self.streaming_half, wrists)
        encoded = torch.cat(
            [torch.cat(encoded_wrists.split(1, dim=0), dim=-1), encoded_high],
            dim=-2,
        )
        mu, _logvar = torch.chunk(encoded, 2, dim=1)
        mean = torch.tensor(self.vae.config.latents_mean, device=mu.device).view(
            1, -1, 1, 1, 1
        )
        std = torch.tensor(self.vae.config.latents_std, device=mu.device).view(
            1, -1, 1, 1, 1
        )
        normalized = ((mu.float() - mean) / std).to(mu.dtype)
        return torch.cat(normalized.split(1, dim=0), dim=-1).to(self.device)

