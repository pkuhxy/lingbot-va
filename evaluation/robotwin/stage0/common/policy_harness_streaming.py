"""Policy harness with 1+4-frame streaming observation encoding."""

from __future__ import annotations

from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

from .observation_encoder_chunked import ChunkedRobotwinObservationEncoder
from .policy_harness import DeterministicPolicyHarness


def _encode_obs_chunked(server, obs):
    images = obs["obs"]
    if not isinstance(images, list):
        images = [images]
    if not images:
        return None
    videos = []
    for camera_index, key in enumerate(server.job_config.obs_cam_keys):
        if server.env_type == "robotwin_tshape":
            if camera_index == 0:
                height, width = server.height, server.width
            else:
                height, width = server.height // 2, server.width // 2
        else:
            height, width = server.height, server.width
        video = torch.from_numpy(
            np.stack([frame[key] for frame in images])
        ).float().permute(3, 0, 1, 2)
        video = F.interpolate(
            video, size=(height, width), mode="bilinear", align_corners=False
        ).unsqueeze(0)
        videos.append(video)

    if server.env_type == "robotwin_tshape":
        high = (videos[0] / 255.0 * 2.0 - 1.0).to(server.device, server.dtype)
        wrists = (
            torch.cat(videos[1:], dim=0) / 255.0 * 2.0 - 1.0
        ).to(server.device, server.dtype)
        encoded_high = ChunkedRobotwinObservationEncoder._encode_stream(
            server.streaming_vae, high
        )
        encoded_wrists = ChunkedRobotwinObservationEncoder._encode_stream(
            server.streaming_vae_half, wrists
        )
        encoded = torch.cat(
            [torch.cat(encoded_wrists.split(1, dim=0), dim=-1), encoded_high],
            dim=-2,
        )
    else:
        video = (torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0).to(
            server.device, server.dtype
        )
        encoded = ChunkedRobotwinObservationEncoder._encode_stream(
            server.streaming_vae, video
        )
    mu, _logvar = torch.chunk(encoded, 2, dim=1)
    mean = torch.tensor(server.vae.config.latents_mean, device=mu.device)
    std = torch.tensor(server.vae.config.latents_std, device=mu.device)
    normalized = server.normalize_latents(mu, mean, 1.0 / std)
    return torch.cat(normalized.split(1, dim=0), dim=-1).to(server.device)


class StreamingDeterministicPolicyHarness(DeterministicPolicyHarness):
    def __init__(self, server, global_seed: int, negative_prompt: str = ""):
        super().__init__(server, global_seed, negative_prompt)
        server._encode_obs = MethodType(_encode_obs_chunked, server)

