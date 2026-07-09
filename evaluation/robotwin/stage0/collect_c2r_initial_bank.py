"""Collect one expert-validated initial observation per task and C2R split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.config import load_config, torch_dtype
from common.data import save_rgb_npz
from common.io import read_jsonl, sample_storage_name, write_jsonl
from common.observation_encoder import RobotwinObservationEncoder
from common.prompt_encoder import RobotwinPromptEncoder
from collect_c2r_observation_bank import cameras_from_obs, eef_state


def balanced_records(paths: list[Path], max_per_split: int):
    selected = []
    for path in paths:
        seen_tasks = set()
        count = 0
        for record in read_jsonl(path):
            task = record["task"]
            if task in seen_tasks:
                continue
            if not record.get("expert_validated", False):
                raise ValueError(f"C2R record is not expert-validated: {record}")
            selected.append(record)
            seen_tasks.add(task)
            count += 1
            if count >= max_per_split:
                break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validated-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path)
    parser.add_argument("--max-per-split", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir.expanduser().resolve()
    rows = balanced_records(args.validated_manifest, args.max_per_split)

    from evaluation.robotwin.generate_rt_c2r_validated_manifests import (
        build_task_args, class_decorator, close_env_quietly,
        configure_robotwin_imports, robotwin_root_from_env,
    )

    robotwin_root = (
        args.robotwin_root.expanduser().resolve()
        if args.robotwin_root else robotwin_root_from_env()
    )
    configure_robotwin_imports(robotwin_root)
    dtype = torch_dtype(config["runtime"].get("dtype", "bfloat16"))
    device = config["runtime"].get("device", "cuda:0")
    encoder = RobotwinObservationEncoder(
        config["model_root"], device=device, dtype=dtype
    )
    prompt_encoder = RobotwinPromptEncoder(
        config["model_root"], device=device, dtype=dtype
    )

    output = []
    for index, record in enumerate(rows):
        task, split, seed = record["task"], record["split"], int(record["seed"])
        environment = class_decorator(task)
        task_args = build_task_args(robotwin_root, task, split)
        try:
            environment.setup_demo(
                now_ep_num=int(record.get("episode_id", 0)),
                seed=seed, is_test=True, **task_args,
            )
            observation = environment.get_obs()
        finally:
            close_env_quietly(environment)

        prompt = record.get("prompt") or task.replace("_", " ")
        cameras = cameras_from_obs(observation)
        sample_id = f"c2r/{split}/{task}/seed_{seed}/progress_000"
        name = sample_storage_name(sample_id)
        rgb_path = output_dir / "rgb" / f"{name}.npz"
        state_path = output_dir / "state" / f"{name}.npy"
        tensor_path = output_dir / "tensors" / f"{name}.pt"
        save_rgb_npz(rgb_path, cameras)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np
        np.save(state_path, eef_state(observation))

        latent = encoder.encode(cameras, reset=True).float().cpu().squeeze(0)
        text_emb = prompt_encoder.encode(prompt)
        frames = latent.shape[1]
        payload = {
            "sample_id": sample_id,
            "latents": latent,
            "actions": torch.zeros(30, frames, 16, 1),
            "actions_mask": torch.zeros(30, frames, 16, 1, dtype=torch.bool),
            "text_emb": text_emb,
            "variants": {},
        }
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, tensor_path)
        output.append({
            "sample_id": sample_id,
            "domain": "c2r",
            "split": split,
            "c2r_split": split,
            "task": task,
            "seed": seed,
            "episode_or_seed": f"seed_{seed}",
            "progress_bin": 0.0,
            "observation_type": "initial",
            "expert_validated": True,
            "prompt": prompt,
            "rgb_path": str(rgb_path),
            "eef_state_path": str(state_path),
            "tensor_path": str(tensor_path),
        })
        print(f"[{index + 1}/{len(rows)}] {split}/{task}/{seed}", flush=True)

    manifest = output_dir / "manifest.jsonl"
    write_jsonl(manifest, output)
    print(f"Wrote {len(output)} C2R initial observations: {manifest}")


if __name__ == "__main__":
    main()
