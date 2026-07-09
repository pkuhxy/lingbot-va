"""Estimate centered directional finite differences of train-forward action velocity."""

from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from pathlib import Path

import torch

from common.checkpoint import load_transformer_checkpoint
from common.config import checkpoint_items, load_config, torch_dtype
from common.data import action_velocity, load_tensor_sample
from common.deterministic import DeterministicInputBuilder
from common.io import get_run_dir, metric_row, read_jsonl, write_csv


def masked_norm(value, mask):
    return value.float()[torch.broadcast_to(mask.bool(), value.shape)].norm()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = get_run_dir(config, args.run_dir)
    records = list(read_jsonl(args.bank_manifest))
    records = records[:args.max_samples] if args.max_samples is not None else records
    runtime, forward = config["runtime"], config["train_forward"]
    device, dtype = torch.device(runtime.get("device", "cuda:0")), torch_dtype(runtime.get("dtype", "bfloat16"))
    rows = []
    for checkpoint, path in checkpoint_items(config):
        model = load_transformer_checkpoint(path, torch.float32, "cpu", runtime.get("attn_mode", "flex")).to(device=device, dtype=dtype).eval().requires_grad_(False)
        builder = DeterministicInputBuilder(runtime["seed"], patch_size=tuple(model.patch_size))
        with torch.inference_mode():
            for record in records:
                clean = load_tensor_sample(record, "clean", device)
                noises = builder.noises(clean, record["sample_id"])
                values = defaultdict(list)
                for pair in record.get("directional_pairs", []):
                    for quantile in forward["timestep_quantiles"]:
                        velocities = {}
                        metadata = None
                        for side in ("plus", "minus"):
                            sample = load_tensor_sample(record, pair[side], device)
                            input_dict, metadata = builder.build_train_input(
                                sample, record["sample_id"], quantile,
                                forward["chunk_size"], forward["window_size"], noises=noises,
                            )
                            velocities[side] = action_velocity(model(input_dict, train_mode=True), sample["actions"].shape[2])
                        jacobian = float(masked_norm(velocities["plus"] - velocities["minus"], clean["actions_mask"]) / (2 * pair["epsilon"]))
                        group = "nuisance" if pair["direction"] in {"photo", "background"} else "signal"
                        values[(quantile, pair["epsilon"], group)].append(jacobian)
                        rows.append(metric_row(
                            run_id=run_dir.name, checkpoint=checkpoint, sample_id=record["sample_id"],
                            task=record.get("task", ""), episode_or_seed=record.get("episode_or_seed", record.get("episode_index", "")),
                            domain="clean", perturbation=f"direction_{pair['direction']}", severity=pair["epsilon"],
                            timestep_quantile=quantile, timestep=metadata["action"]["timestep"], sigma=metadata["action"]["sigma"],
                            noise_seed=metadata["action_noise_seed"], token_group="action_velocity",
                            metric_name="centered_directional_fd", metric_value=jacobian, direction_group=group,
                        ))
                for (quantile, epsilon, _group) in sorted(values):
                    nuisance = values.get((quantile, epsilon, "nuisance"), [])
                    signal = values.get((quantile, epsilon, "signal"), [])
                    if nuisance and signal:
                        selectivity = (sum(signal) / len(signal)) / max(sum(nuisance) / len(nuisance), 1e-12)
                        rows.append(metric_row(
                            run_id=run_dir.name, checkpoint=checkpoint, sample_id=record["sample_id"], task=record.get("task", ""),
                            episode_or_seed=record.get("episode_or_seed", record.get("episode_index", "")), domain="clean",
                            severity=epsilon, timestep_quantile=quantile, token_group="action_velocity",
                            metric_name="directional_selectivity", metric_value=selectivity,
                        ))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    output = run_dir / "metrics" / "directional_fd.csv"
    write_csv(output, rows)
    print(f"Wrote {len(rows)} metrics: {output}")


if __name__ == "__main__":
    main()

