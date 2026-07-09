from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch


def stable_seed(global_seed: int, sample_id: str, stream_name: str) -> int:
    payload = f"{global_seed}\0{sample_id}\0{stream_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def generator_for(global_seed: int, sample_id: str, stream_name: str) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(global_seed, sample_id, stream_name) % (2**63 - 1))
    return generator


def deterministic_noise(
    reference: torch.Tensor,
    global_seed: int,
    sample_id: str,
    stream_name: str,
) -> torch.Tensor:
    noise = torch.randn(
        reference.shape,
        generator=generator_for(global_seed, sample_id, stream_name),
        dtype=torch.float32,
        device="cpu",
    )
    return noise.to(device=reference.device, dtype=reference.dtype)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class TimestepChoice:
    quantile: float
    index: int
    timestep: float
    sigma: float


class DeterministicInputBuilder:
    def __init__(
        self,
        global_seed: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        video_shift: float = 5.0,
        action_shift: float = 1.0,
    ) -> None:
        from wan_va.utils import FlowMatchScheduler

        self.global_seed = int(global_seed)
        self.patch_size = tuple(patch_size)
        self.video_scheduler = FlowMatchScheduler(
            shift=video_shift, sigma_min=0.0, extra_one_step=True
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=action_shift, sigma_min=0.0, extra_one_step=True
        )
        self.video_scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

    @staticmethod
    def choice(scheduler, quantile: float) -> TimestepChoice:
        quantile = float(quantile)
        index = int(round(quantile * (len(scheduler.timesteps) - 1)))
        return TimestepChoice(
            quantile, index, float(scheduler.timesteps[index].item()),
            float(scheduler.sigmas[index].item()),
        )

    def noises(self, batch: dict[str, torch.Tensor], sample_id: str) -> dict[str, torch.Tensor]:
        return {
            "video": deterministic_noise(
                batch["latents"], self.global_seed, sample_id, "video_noise"
            ),
            "action": deterministic_noise(
                batch["actions"], self.global_seed, sample_id, "action_noise"
            ),
        }

    def _branch(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor,
        scheduler,
        choice: TimestepChoice,
        action_mode: bool,
        action_mask: torch.Tensor | None = None,
        condition_noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        from wan_va.utils import get_mesh_id

        batch, _, frames, height, width = latent.shape
        ids = torch.full((frames,), choice.index, dtype=torch.long, device="cpu")
        timesteps = scheduler.timesteps[ids].to(latent.device)
        noisy = scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = scheduler.training_target(latent, noise, timesteps)
        cond_timesteps = torch.zeros_like(timesteps)
        condition = latent
        if condition_noise is not None:
            condition = scheduler.add_noise(latent, condition_noise, timesteps, t_dim=2)
            cond_timesteps = timesteps.clone()
        if action_mask is not None:
            mask = action_mask.to(device=latent.device, dtype=latent.dtype)
            noisy, targets, condition = noisy * mask, targets * mask, condition * mask
        patch = (1, 1, 1) if action_mode else self.patch_size
        grid = get_mesh_id(
            frames // patch[0], height // patch[1], width // patch[2],
            t=1 if action_mode else 0, f_w=1, f_shift=0, action=action_mode,
        ).to(latent.device)
        return {
            "timesteps": timesteps[None].repeat(batch, 1),
            "noisy_latents": noisy, "targets": targets, "latent": condition,
            "cond_timesteps": cond_timesteps[None].repeat(batch, 1),
            "grid_id": grid[None].repeat(batch, 1, 1),
        }

    def build_train_input(
        self,
        batch: dict[str, torch.Tensor],
        sample_id: str,
        quantile: float,
        chunk_size: int = 2,
        window_size: int = 64,
        noises: dict[str, torch.Tensor] | None = None,
        condition_video_noise: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        required = {"latents", "actions", "actions_mask", "text_emb"}
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"Sample {sample_id} is missing tensors: {sorted(missing)}")
        if batch["latents"].shape[0] != 1:
            raise ValueError("DeterministicInputBuilder currently requires batch size 1")
        noises = noises or self.noises(batch, sample_id)
        video_choice = self.choice(self.video_scheduler, quantile)
        action_choice = self.choice(self.action_scheduler, quantile)
        cond_noise = None
        if condition_video_noise:
            cond_noise = deterministic_noise(
                batch["latents"], self.global_seed, sample_id, "condition_video_noise"
            )
        latent_dict = self._branch(
            batch["latents"], noises["video"], self.video_scheduler,
            video_choice, False, condition_noise=cond_noise,
        )
        action_dict = self._branch(
            batch["actions"], noises["action"], self.action_scheduler,
            action_choice, True, action_mask=batch["actions_mask"],
        )
        latent_dict["text_emb"] = batch["text_emb"]
        action_dict["text_emb"] = batch["text_emb"]
        action_dict["actions_mask"] = batch["actions_mask"]
        value = {
            "latent_dict": latent_dict, "action_dict": action_dict,
            "chunk_size": int(chunk_size), "window_size": int(window_size),
        }
        metadata = {
            "sample_id": sample_id, "video": video_choice.__dict__,
            "action": action_choice.__dict__,
            "video_noise_seed": stable_seed(self.global_seed, sample_id, "video_noise"),
            "action_noise_seed": stable_seed(self.global_seed, sample_id, "action_noise"),
            "video_noise_sha256": tensor_sha256(noises["video"]),
            "action_noise_sha256": tensor_sha256(noises["action"]),
            "chunk_size": int(chunk_size), "window_size": int(window_size),
        }
        return value, metadata


def move_batch(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

