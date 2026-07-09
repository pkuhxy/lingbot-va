"""Collect policy-independent initial/final observations from validated C2R expert seeds."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from common.config import load_config, torch_dtype
from common.data import save_rgb_npz
from common.io import read_jsonl, sample_storage_name, write_jsonl
from common.observation_encoder import CAMERA_KEYS, RobotwinObservationEncoder
from common.prompt_encoder import RobotwinPromptEncoder


def cameras_from_obs(observation):
    cameras = observation["observation"]
    return {
        CAMERA_KEYS[0]: np.asarray(cameras["head_camera"]["rgb"])[None],
        CAMERA_KEYS[1]: np.asarray(cameras["left_camera"]["rgb"])[None],
        CAMERA_KEYS[2]: np.asarray(cameras["right_camera"]["rgb"])[None],
    }


def eef_state(observation):
    return np.asarray(
        observation["endpose"]["left_endpose"]
        + [observation["endpose"]["left_gripper"]]
        + observation["endpose"]["right_endpose"]
        + [observation["endpose"]["right_gripper"]],
        dtype=np.float32,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validated-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, default=None)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--skip-encoding", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir.expanduser().resolve()
    rows = [row for path in args.validated_manifest for row in read_jsonl(path)]
    rows = rows[:args.max_records] if args.max_records is not None else rows

    from evaluation.robotwin.generate_rt_c2r_validated_manifests import (
        build_task_args, class_decorator, close_env_quietly, configure_robotwin_imports,
        robotwin_root_from_env,
    )
    robotwin_root = args.robotwin_root.expanduser().resolve() if args.robotwin_root else robotwin_root_from_env()
    configure_robotwin_imports(robotwin_root)
    encoder = prompt_encoder = None
    if not args.skip_encoding:
        dtype = torch_dtype(config["runtime"].get("dtype", "bfloat16"))
        device = config["runtime"].get("device", "cuda:0")
        encoder = RobotwinObservationEncoder(config["model_root"], device=device, dtype=dtype)
        prompt_encoder = RobotwinPromptEncoder(config["model_root"], device=device, dtype=dtype)
    output = []
    for record in rows:
        task, split, seed = record["task"], record["split"], int(record["seed"])
        env = class_decorator(task)
        task_args = build_task_args(robotwin_root, task, split)
        try:
            env.setup_demo(now_ep_num=int(record.get("episode_id", 0)), seed=seed, is_test=True, **task_args)
            initial = env.get_obs()
            env.play_once()
            final = env.get_obs()
            valid = bool(env.plan_success and env.check_success())
        finally:
            close_env_quietly(env)
        if not valid:
            raise RuntimeError(f"Validated seed no longer passes expert: {split}/{task}/{seed}")
        prompt = record.get("prompt") or task.replace("_", " ")
        text_emb = prompt_encoder.encode(prompt) if prompt_encoder else None
        for progress, observation, observation_type in (
            (0.0, initial, "initial"), (1.0, final, "final")
        ):
            sample_id = f"c2r/{split}/{task}/seed_{seed}/progress_{int(progress*100):03d}"
            name = sample_storage_name(sample_id)
            cameras = cameras_from_obs(observation)
            rgb_path = output_dir / "rgb" / f"{name}.npz"
            state_path = output_dir / "state" / f"{name}.npy"
            save_rgb_npz(rgb_path, cameras)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(state_path, eef_state(observation))
            tensor_path = None
            if encoder is not None:
                latent = encoder.encode(cameras, reset=True).float().cpu().squeeze(0)
                frames = latent.shape[1]
                payload = {
                    "sample_id": sample_id, "latents": latent,
                    "actions": torch.zeros(30, frames, 16, 1),
                    "actions_mask": torch.zeros(30, frames, 16, 1, dtype=torch.bool),
                    "text_emb": text_emb, "variants": {},
                }
                tensor_path = output_dir / "tensors" / f"{name}.pt"
                tensor_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, tensor_path)
            output.append({
                "sample_id": sample_id, "domain": "c2r", "split": split,
                "c2r_split": split, "task": task, "seed": seed,
                "episode_or_seed": f"seed_{seed}", "progress_bin": progress,
                "observation_type": observation_type, "expert_validated": True,
                "prompt": prompt, "rgb_path": str(rgb_path),
                "eef_state_path": str(state_path),
                "tensor_path": str(tensor_path) if tensor_path else None,
            })
        print(f"[{len(output)//2}/{len(rows)}] {split}/{task}/{seed}")
    manifest = output_dir / "manifest.jsonl"
    write_jsonl(manifest, output)
    print(f"Wrote {len(output)} C2R observations: {manifest}")


if __name__ == "__main__":
    main()

