from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from .config import torch_dtype
from .deterministic import deterministic_noise, stable_seed, tensor_sha256
from .observation_encoder import CAMERA_KEYS


class DeterministicPolicyHarness:
    """Runs the server's video-then-action path with explicit paired noise."""

    def __init__(self, server, global_seed: int, negative_prompt: str = ""):
        self.server = server
        self.global_seed = int(global_seed)
        self.negative_prompt = negative_prompt

    @classmethod
    def from_stage0_config(cls, config: dict[str, Any], transformer_path: str | Path):
        from wan_va.configs import VA_CONFIGS
        from wan_va.wan_va_server import VA_Server

        job = copy.deepcopy(VA_CONFIGS["robotwin"])
        runtime, inference = config["runtime"], config["policy_inference"]
        job.local_rank = int(str(runtime.get("device", "cuda:0")).split(":")[-1])
        job.rank, job.world_size = 0, 1
        job.param_dtype = torch_dtype(runtime.get("dtype", "bfloat16"))
        job.transformer_model_name_or_path = str(Path(transformer_path).resolve())
        job.wan22_pretrained_model_name_or_path = str(Path(config["model_root"]).resolve())
        job.num_inference_steps = int(inference.get("video_steps", 25))
        job.action_num_inference_steps = int(inference.get("action_steps", 50))
        job.guidance_scale = float(inference.get("guidance_scale", 5.0))
        job.action_guidance_scale = float(inference.get("action_guidance_scale", 1.0))
        job.enable_offload = False
        job.save_root = str(Path(config["output_dir"]) / "policy_harness_cache")
        server = VA_Server(job)
        return cls(server, config["runtime"]["seed"], inference.get("negative_prompt", ""))

    @staticmethod
    def server_observation(cameras: dict[str, np.ndarray]) -> dict[str, Any]:
        frames = len(cameras[CAMERA_KEYS[0]])
        if any(len(cameras[key]) != frames for key in CAMERA_KEYS):
            raise ValueError("All camera clips must have equal frame counts")
        return {"obs": [{key: cameras[key][index] for key in CAMERA_KEYS} for index in range(frames)]}

    def _set_prompt(self, prompt: str) -> None:
        server = self.server
        server._reset(prompt=prompt)
        if server.use_cfg:
            server.prompt_embeds, server.negative_prompt_embeds = server.encode_prompt(
                prompt=prompt, negative_prompt=self.negative_prompt,
                do_classifier_free_guidance=True, max_sequence_length=512,
                device=server.device, dtype=server.dtype,
            )

    @torch.no_grad()
    def run(self, cameras: dict[str, np.ndarray], prompt: str, sample_id: str) -> dict[str, Any]:
        from wan_va.utils import data_seq_to_patch

        server = self.server
        self._set_prompt(prompt)
        observation = self.server_observation(cameras)
        current = server._encode_obs(observation)
        frame_count = server.job_config.frame_chunk_size
        video_ref = torch.empty(
            1, 48, frame_count, server.latent_height, server.latent_width,
            device=server.device, dtype=server.dtype,
        )
        action_ref = torch.empty(
            1, server.job_config.action_dim, frame_count,
            server.action_per_frame, 1, device=server.device, dtype=server.dtype,
        )
        video_noise = deterministic_noise(
            video_ref, self.global_seed, sample_id, "policy_video_noise"
        )
        action_noise = deterministic_noise(
            action_ref, self.global_seed, sample_id, "policy_action_noise"
        )
        latents, actions = video_noise.clone(), action_noise.clone()
        server.scheduler.set_timesteps(server.job_config.num_inference_steps)
        server.action_scheduler.set_timesteps(server.job_config.action_num_inference_steps)
        video_timesteps = F.pad(server.scheduler.timesteps, (0, 1), value=0)
        action_timesteps = F.pad(server.action_scheduler.timesteps, (0, 1), value=0)
        action_velocities = []

        for index, timestep in enumerate(video_timesteps):
            last = index == len(video_timesteps) - 1
            condition = current[:, :, 0:1].to(server.dtype)
            input_dict = server._prepare_latent_input(
                latents, None, timestep, timestep, condition, None, frame_st_id=0
            )["latent_res_lst"]
            velocity = server.transformer(
                server._repeat_input_for_cfg(input_dict), update_cache=1 if last else 0,
                cache_name=server.cache_name, action_mode=False,
            )
            if not last:
                velocity = data_seq_to_patch(
                    server.job_config.patch_size, velocity, frame_count,
                    server.latent_height, server.latent_width,
                    batch_size=2 if server.use_cfg else 1,
                )
                if server.job_config.guidance_scale > 1:
                    velocity = velocity[1:] + server.job_config.guidance_scale * (velocity[:1] - velocity[1:])
                else:
                    velocity = velocity[:1]
                latents = server.scheduler.step(velocity, timestep, latents, return_dict=False)
            latents[:, :, 0:1] = condition

        for index, timestep in enumerate(action_timesteps):
            last = index == len(action_timesteps) - 1
            condition = torch.zeros(
                1, server.job_config.action_dim, 1, server.action_per_frame, 1,
                device=server.device, dtype=server.dtype,
            )
            input_dict = server._prepare_latent_input(
                None, actions, timestep, timestep, None, condition, frame_st_id=0
            )["action_res_lst"]
            velocity = server.transformer(
                server._repeat_input_for_cfg(input_dict), update_cache=1 if last else 0,
                cache_name=server.cache_name, action_mode=True,
            )
            if not last:
                velocity = rearrange(velocity, "b (f n) c -> b c f n 1", f=frame_count)
                if server.job_config.action_guidance_scale > 1:
                    velocity = velocity[1:] + server.job_config.action_guidance_scale * (velocity[:1] - velocity[1:])
                else:
                    velocity = velocity[:1]
                action_velocities.append(velocity.float().cpu())
                actions = server.action_scheduler.step(velocity, timestep, actions, return_dict=False)
            actions[:, :, 0:1] = condition
        actions[:, ~server.action_mask] *= 0
        physical = server.postprocess_action(actions.clone())
        return {
            "action": torch.from_numpy(physical),
            "action_normalized": actions.float().cpu(),
            "future_latent": latents.float().cpu(),
            "current_latent": current.float().cpu(),
            "action_velocities": torch.stack(action_velocities),
            "metadata": {
                "sample_id": sample_id,
                "video_noise_seed": stable_seed(self.global_seed, sample_id, "policy_video_noise"),
                "action_noise_seed": stable_seed(self.global_seed, sample_id, "policy_action_noise"),
                "video_noise_sha256": tensor_sha256(video_noise),
                "action_noise_sha256": tensor_sha256(action_noise),
            },
        }

    def run_pair(self, clean, augmented, prompt: str, sample_id: str):
        clean_result = self.run(clean, prompt, sample_id)
        augmented_result = self.run(augmented, prompt, sample_id)
        for key in ("video_noise_sha256", "action_noise_sha256"):
            if clean_result["metadata"][key] != augmented_result["metadata"][key]:
                raise RuntimeError(f"Paired policy noise mismatch for {sample_id}: {key}")
        return clean_result, augmented_result

    def close(self) -> None:
        server = self.server
        server.transformer.clear_cache(server.cache_name)
        server.streaming_vae.clear_cache()
        if server.streaming_vae_half is not None:
            server.streaming_vae_half.clear_cache()

