"""Encode clean and paired RGB perturbations through the server-equivalent VAE."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from common.config import load_config, torch_dtype
from common.data import save_rgb_npz, variant_name
from common.deterministic import stable_seed
from common.io import read_jsonl, sample_storage_name, write_jsonl
from common.observation_encoder import CAMERA_KEYS, RobotwinObservationEncoder
from common.perturbations import configured_perturbations, directional_pair, perturb_cameras


def read_video_window(path: str, start: int, end: int) -> np.ndarray:
    frames = []
    for index, frame in enumerate(iio.imiter(path, plugin="pyav")):
        if index >= end:
            break
        if index >= start:
            frames.append(np.asarray(frame)[..., :3])
    if len(frames) != end - start:
        raise ValueError(f"Expected {end-start} frames from {path}, got {len(frames)}")
    return np.stack(frames)


class TrainingSampleLoader:
    def __init__(self):
        from wan_va.configs import VA_CONFIGS

        self.config = copy.deepcopy(VA_CONFIGS["robotwin_train"])
        self.config.cfg_prob = 0.0
        self.datasets = {}

    def __call__(self, record):
        from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset

        repo = record["repo_id"]
        if repo not in self.datasets:
            self.datasets[repo] = LatentLeRobotDataset(repo_id=repo, config=self.config)
        dataset = self.datasets[repo]
        wanted = (record["episode_index"], record["start_frame"], record["end_frame"])
        for index, meta in enumerate(dataset.new_metas):
            found = (meta["episode_index"], meta["start_frame"], meta["end_frame"])
            if found == wanted:
                return dataset[index]
        raise KeyError(f"Training latent window not found: {wanted} in {repo}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--require-latent-match", action="store_true")
    parser.add_argument("--skip-directional", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = args.manifest or Path(config["clean_manifest"])
    records = list(read_jsonl(manifest))
    if args.max_samples is not None:
        records = records[:args.max_samples]
    device = config["runtime"].get("device", "cuda:0")
    encoder = RobotwinObservationEncoder(
        config["model_root"], device=device,
        dtype=torch_dtype(config["runtime"].get("dtype", "bfloat16")),
    )
    load_training = TrainingSampleLoader()
    output_dir = args.output_dir.expanduser().resolve()
    bank_rows = []
    for record in records:
        sample_id = record["sample_id"]
        name = sample_storage_name(sample_id)
        cameras = {
            key: read_video_window(record["video_paths"][key], record["start_frame"], record["end_frame"])
            for key in CAMERA_KEYS
        }
        training = load_training(record)
        clean_latent = encoder.encode(cameras, reset=True).float().cpu()
        preencoded = training["latents"][None].float().cpu()
        identity_error = None
        if clean_latent.shape == preencoded.shape:
            identity_error = float((clean_latent - preencoded).norm() / preencoded.norm().clamp_min(1e-12))
        elif args.require_latent_match:
            raise ValueError(f"Clean latent shape mismatch for {sample_id}: {clean_latent.shape} vs {preencoded.shape}")
        if args.require_latent_match and identity_error is not None and identity_error > 0.05:
            raise ValueError(f"Clean latent identity error {identity_error:.4g} exceeds 0.05: {sample_id}")
        payload = {
            "sample_id": sample_id, "latents": clean_latent.squeeze(0),
            "actions": training["actions"].cpu(),
            "actions_mask": training["actions_mask"].cpu(),
            "text_emb": training["text_emb"].cpu(), "variants": {},
        }
        rgb_dir = output_dir / "rgb" / name
        clean_rgb_path = rgb_dir / "clean.npz"
        save_rgb_npz(clean_rgb_path, cameras)

        # Re-encode unchanged pixels as the mandatory cache/streaming identity test.
        identity = encoder.encode(cameras, reset=True).float().cpu()
        payload["variants"]["identity"] = {"latents": identity.squeeze(0), "severity": 0.0}
        save_rgb_npz(rgb_dir / "identity.npz", cameras)
        variants = {"identity": {"rgb_path": str(rgb_dir / "identity.npz"), "severity": 0.0}}
        for perturbation, severity in configured_perturbations(config["perturbations"]):
            transformed, transform_seed = perturb_cameras(
                cameras, perturbation, severity, config["runtime"]["seed"], sample_id
            )
            variant = variant_name(perturbation, severity)
            latent = encoder.encode(transformed, reset=True).float().cpu()
            payload["variants"][variant] = {
                "latents": latent.squeeze(0), "perturbation": perturbation,
                "severity": severity, "transform_seed": transform_seed,
            }
            rgb_path = rgb_dir / f"{variant}.npz"
            save_rgb_npz(rgb_path, transformed)
            variants[variant] = {
                "rgb_path": str(rgb_path), "perturbation": perturbation,
                "severity": severity, "transform_seed": transform_seed,
            }
        directional = []
        if not args.skip_directional:
            for kind in ("photo", "geometry"):
                for epsilon in config["perturbations"].get("directional_epsilons", [0.5, 1.0]):
                    plus, minus = {}, {}
                    seed = stable_seed(config["runtime"]["seed"], sample_id, f"direction:{kind}:{epsilon}")
                    for camera, images in cameras.items():
                        plus[camera], minus[camera] = directional_pair(images, kind, float(epsilon), seed)
                    pair = {}
                    for side, transformed in (("plus", plus), ("minus", minus)):
                        variant = variant_name(f"direction_{kind}_{side}", float(epsilon))
                        latent = encoder.encode(transformed, reset=True).float().cpu()
                        payload["variants"][variant] = {
                            "latents": latent.squeeze(0), "direction": kind,
                            "epsilon": float(epsilon), "side": side, "transform_seed": seed,
                        }
                        rgb_path = rgb_dir / f"{variant}.npz"
                        save_rgb_npz(rgb_path, transformed)
                        variants[variant] = {"rgb_path": str(rgb_path), **payload["variants"][variant] | {"latents": None}}
                        variants[variant].pop("latents", None)
                        pair[side] = variant
                    directional.append({"direction": kind, "epsilon": float(epsilon), **pair})
        tensor_path = output_dir / "tensors" / f"{name}.pt"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, tensor_path)
        bank_rows.append({
            **record, "tensor_path": str(tensor_path), "rgb_path": str(clean_rgb_path),
            "variants": variants, "directional_pairs": directional,
            "preencoded_identity_relative_l2": identity_error,
        })
        print(f"[{len(bank_rows)}/{len(records)}] {sample_id}")
    output_manifest = output_dir / "manifest.jsonl"
    write_jsonl(output_manifest, bank_rows)
    print(f"Wrote paired bank with {len(bank_rows)} samples: {output_manifest}")


if __name__ == "__main__":
    main()

