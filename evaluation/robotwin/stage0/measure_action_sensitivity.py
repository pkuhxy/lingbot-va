"""Measure paired train-forward or full-policy action sensitivity."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

from common.checkpoint import load_transformer_checkpoint
from common.config import checkpoint_items, load_config, torch_dtype
from common.data import action_velocity, load_rgb_npz, load_tensor_sample
from common.deterministic import DeterministicInputBuilder
from common.io import get_run_dir, metric_row, read_jsonl, sample_storage_name, write_csv
from common.metrics import relative_l2
from common.policy_harness import DeterministicPolicyHarness


CHANNEL_GROUPS = {
    "left_position": list(range(0, 3)), "left_rotation": list(range(3, 7)),
    "right_position": list(range(7, 10)), "right_rotation": list(range(10, 14)),
    "gripper": [28, 29],
}


def variants(record):
    for name, metadata in record.get("variants", {}).items():
        if name == "identity" or "perturbation" in metadata:
            yield name, metadata


def train_rows(config, run_dir, records):
    rows = []
    runtime, forward = config["runtime"], config["train_forward"]
    device, dtype = torch.device(runtime.get("device", "cuda:0")), torch_dtype(runtime.get("dtype", "bfloat16"))
    for checkpoint, path in checkpoint_items(config):
        model = load_transformer_checkpoint(
            path, torch.float32, "cpu", runtime.get("attn_mode", "flex")
        ).to(device=device, dtype=dtype).eval().requires_grad_(False)
        builder = DeterministicInputBuilder(runtime["seed"], patch_size=tuple(model.patch_size))
        with torch.inference_mode():
            for record in records:
                sample_id = record["sample_id"]
                clean = load_tensor_sample(record, "clean", device)
                shared_noise = builder.noises(clean, sample_id)
                for quantile in forward["timestep_quantiles"]:
                    clean_input, meta = builder.build_train_input(
                        clean, sample_id, quantile, forward["chunk_size"],
                        forward["window_size"], noises=shared_noise,
                    )
                    clean_velocity = action_velocity(model(clean_input, train_mode=True), clean["actions"].shape[2])
                    for variant, variant_meta in variants(record):
                        augmented = load_tensor_sample(record, variant, device)
                        aug_input, aug_meta = builder.build_train_input(
                            augmented, sample_id, quantile, forward["chunk_size"],
                            forward["window_size"], noises=shared_noise,
                        )
                        if meta["action_noise_sha256"] != aug_meta["action_noise_sha256"]:
                            raise RuntimeError(f"Action noise mismatch for {sample_id}/{variant}")
                        aug_velocity = action_velocity(model(aug_input, train_mode=True), augmented["actions"].shape[2])
                        mask = clean["actions_mask"]
                        common = dict(
                            run_id=run_dir.name, checkpoint=checkpoint, sample_id=sample_id,
                            task=record.get("task", ""), episode_or_seed=record.get("episode_or_seed", record.get("episode_index", "")),
                            domain="clean", perturbation=variant_meta.get("perturbation", "identity"),
                            severity=variant_meta.get("severity", 0), timestep_quantile=quantile,
                            timestep=meta["action"]["timestep"], sigma=meta["action"]["sigma"],
                            noise_seed=meta["action_noise_seed"], transform_seed=variant_meta.get("transform_seed", ""),
                            token_group="action_velocity",
                        )
                        rows.append(metric_row(**common, metric_name="velocity_relative_l2", metric_value=relative_l2(clean_velocity, aug_velocity, mask)))
                        for group, channels in CHANNEL_GROUPS.items():
                            rows.append(metric_row(
                                **common, metric_name=f"velocity_relative_l2/{group}",
                                metric_value=relative_l2(clean_velocity[:, channels], aug_velocity[:, channels], mask[:, channels]),
                            ))
                        for frame in range(clean_velocity.shape[2]):
                            rows.append(metric_row(
                                **common, metric_name="velocity_relative_l2/frame",
                                metric_value=relative_l2(clean_velocity[:, :, frame], aug_velocity[:, :, frame], mask[:, :, frame]),
                                action_frame=frame,
                            ))
                print(f"{checkpoint}: {sample_id}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def quaternion_geodesic(left, right):
    left, right = F.normalize(left.float(), dim=-1), F.normalize(right.float(), dim=-1)
    return float((2 * torch.acos((left * right).sum(-1).abs().clamp(max=1))).mean())


def policy_metric_rows(clean, augmented, common):
    left, right = clean["action"].float(), augmented["action"].float()
    rows = [
        metric_row(**common, metric_name="action_relative_l2", metric_value=relative_l2(left, right)),
        metric_row(**common, metric_name="action_mean_abs", metric_value=float((right-left).abs().mean())),
    ]
    flat_left, flat_right = left.flatten(1), right.flatten(1)
    quarter = max(1, flat_left.shape[1] // 4)
    rows.append(metric_row(**common, metric_name="action_relative_l2/first_25pct", metric_value=relative_l2(flat_left[:, :quarter], flat_right[:, :quarter])))
    rows.append(metric_row(**common, metric_name="action_relative_l2/last_75pct", metric_value=relative_l2(flat_left[:, quarter:], flat_right[:, quarter:])))
    for group, channels in {"left_position": [0,1,2], "right_position": [8,9,10]}.items():
        rows.append(metric_row(**common, metric_name=f"endpoint_l2/{group}", metric_value=float((flat_right[channels, -1]-flat_left[channels, -1]).norm())))
    for group, channels in {"left_rotation": [3,4,5,6], "right_rotation": [11,12,13,14]}.items():
        rows.append(metric_row(**common, metric_name=f"rotation_geodesic/{group}", metric_value=quaternion_geodesic(flat_left[channels].T, flat_right[channels].T)))
    grippers_left, grippers_right = flat_left[[7, 15]], flat_right[[7, 15]]
    rows.append(metric_row(**common, metric_name="gripper_flip_rate", metric_value=float(((grippers_left > 0.5) != (grippers_right > 0.5)).float().mean())))
    if flat_left.shape[1] >= 3:
        jerk_left = torch.diff(flat_left, n=2, dim=1)
        jerk_right = torch.diff(flat_right, n=2, dim=1)
        rows.append(metric_row(**common, metric_name="temporal_jerk_difference", metric_value=float((jerk_right-jerk_left).abs().mean())))
    return rows


def policy_rows(config, run_dir, records):
    rows = []
    for checkpoint, path in checkpoint_items(config):
        harness = DeterministicPolicyHarness.from_stage0_config(config, path)
        for record in records:
            clean_rgb = load_rgb_npz(record["rgb_path"])
            for variant, variant_meta in variants(record):
                augmented_rgb = load_rgb_npz(variant_meta["rgb_path"])
                clean, augmented = harness.run_pair(clean_rgb, augmented_rgb, record["prompt"], record["sample_id"])
                common = dict(
                    run_id=run_dir.name, checkpoint=checkpoint, sample_id=record["sample_id"],
                    task=record.get("task", ""), episode_or_seed=record.get("episode_or_seed", record.get("episode_index", "")),
                    domain="clean", perturbation=variant_meta.get("perturbation", "identity"),
                    severity=variant_meta.get("severity", 0), transform_seed=variant_meta.get("transform_seed", ""),
                    noise_seed=clean["metadata"]["action_noise_seed"], token_group="final_action",
                )
                rows.extend(policy_metric_rows(clean, augmented, common))
                prediction_path = run_dir / "paired_predictions" / checkpoint / sample_storage_name(record["sample_id"]) / f"{variant}.pt"
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"clean": clean, "augmented": augmented, "record": record, "variant": variant_meta}, prediction_path)
        harness.close()
        del harness
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "policy"), default="train")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = get_run_dir(config, args.run_dir)
    records = list(read_jsonl(args.bank_manifest))
    records = records[:args.max_samples] if args.max_samples is not None else records
    rows = train_rows(config, run_dir, records) if args.mode == "train" else policy_rows(config, run_dir, records)
    output = run_dir / "metrics" / f"action_sensitivity_{args.mode}.csv"
    write_csv(output, rows)
    print(f"Wrote {len(rows)} metrics: {output}")


if __name__ == "__main__":
    main()

