"""Extract deterministic per-layer train-forward features and compute CKA drift."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from common.checkpoint import load_transformer_checkpoint
from common.config import checkpoint_items, load_config, torch_dtype
from common.data import load_tensor_sample
from common.deterministic import DeterministicInputBuilder
from common.feature_hooks import BlockFeatureHooks, automatic_layers, token_layout
from common.io import get_run_dir, metric_row, read_jsonl, write_csv
from common.metrics import cluster_bootstrap, cosine_drift, linear_cka


def parse_layers(raw: str, count: int) -> list[int]:
    return automatic_layers(count) if raw == "auto" else sorted({int(value) for value in raw.split(",")})


def extract_checkpoint(name, path, records, config, output_path, zero_actions=False):
    runtime, forward = config["runtime"], config["train_forward"]
    device = torch.device(runtime.get("device", "cuda:0"))
    dtype = torch_dtype(runtime.get("dtype", "bfloat16"))
    transformer = load_transformer_checkpoint(
        path, dtype=torch.float32, device="cpu", attn_mode=runtime.get("attn_mode", "flex")
    ).to(device=device, dtype=dtype).eval().requires_grad_(False)
    layers = parse_layers(str(forward.get("hook_layers", "auto")), len(transformer.blocks))
    builder = DeterministicInputBuilder(runtime["seed"], patch_size=tuple(transformer.patch_size))
    quantile = float(forward.get("primary_timestep_quantile", 0.5))
    collected = {layer: {} for layer in layers}
    metadata, noise_hashes = [], []
    with torch.inference_mode():
        for index, record in enumerate(records):
            sample = load_tensor_sample(record, device=device)
            if zero_actions:
                sample["actions"].zero_()
                sample["actions_mask"].zero_()
            input_dict, deterministic_meta = builder.build_train_input(
                sample, record["sample_id"], quantile,
                chunk_size=forward.get("chunk_size", 2),
                window_size=forward.get("window_size", 64),
                condition_video_noise=forward.get("condition_video_noise", False),
            )
            layout = token_layout(input_dict, tuple(transformer.patch_size))
            with BlockFeatureHooks(transformer, layers, layout) as hooks:
                transformer(input_dict, train_mode=True)
                pooled = hooks.pooled()
            for layer in layers:
                for group, levels in pooled[layer].items():
                    target = collected[layer].setdefault(group, {"window": [], "frame": []})
                    target["window"].append(levels["window"].squeeze(0))
                    target["frame"].append(levels["frame"].squeeze(0))
            metadata.append({
                "sample_id": record["sample_id"], "task": record.get("task", ""),
                "episode_or_seed": record.get("episode_or_seed", record.get("episode_index", record.get("seed", ""))),
                "domain": record.get("domain", "clean"), "c2r_split": record.get("c2r_split", record.get("split", "")),
                "progress_bin": record.get("progress_bin", ""),
            })
            noise_hashes.append(deterministic_meta)
            print(f"{name}: [{index+1}/{len(records)}] {record['sample_id']}")
    packed = {str(layer): {} for layer in layers}
    for layer in layers:
        for group, levels in collected[layer].items():
            packed[str(layer)][group] = {
                "window": torch.stack(levels["window"]),
                "frame": levels["frame"],
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "checkpoint": name, "checkpoint_path": str(path),
        "sample_ids": [row["sample_id"] for row in metadata],
        "metadata": metadata, "layers": packed, "deterministic_metadata": noise_hashes,
    }, output_path)
    del transformer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def drift_rows(feature_paths, run_id, config):
    features = {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in feature_paths.items()}
    base = features["base"]
    rows = []
    for name, feature in features.items():
        if feature["sample_ids"] != base["sample_ids"]:
            raise ValueError(f"Feature sample order differs for {name}; explicit IDs do not align")
        tasks = [row["task"] for row in base["metadata"]]
        clusters = [str(row["episode_or_seed"]) for row in base["metadata"]]
        for layer in base["layers"]:
            x = base["layers"][layer]["condition_video"]["window"]
            y = feature["layers"][layer]["condition_video"]["window"]
            cka = linear_cka(x, y)
            per_sample = 1.0 - torch.nn.functional.cosine_similarity(x.float(), y.float(), dim=1).numpy()
            ci = cluster_bootstrap(
                per_sample, tasks, clusters,
                repeats=config["statistics"].get("bootstrap_repeats", 2000),
                confidence=config["statistics"].get("confidence_level", 0.95),
                seed=config["runtime"]["seed"],
            )
            common = dict(
                run_id=run_id, checkpoint=name, sample_id="aggregate", task="all",
                episode_or_seed="all", domain="clean", layer=int(layer),
                token_group="condition_video",
            )
            rows.append(metric_row(**common, metric_name="linear_cka", metric_value=cka))
            rows.append(metric_row(**common, metric_name="cosine_drift_mean", metric_value=ci["estimate"], ci_low=ci["ci_low"], ci_high=ci["ci_high"]))
            rows.append(metric_row(**common, metric_name="cosine_drift_median", metric_value=float(np.median(per_sample))))
            for task in sorted(set(tasks)):
                idx = [i for i, value in enumerate(tasks) if value == task]
                if len(idx) >= 2:
                    rows.append(metric_row(
                        **common | {"task": task}, metric_name="task_linear_cka",
                        metric_value=linear_cka(x[idx], y[idx]),
                    ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--feature-set", default="clean")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--zero-actions", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = get_run_dir(config, args.run_dir, args.run_id)
    records = list(read_jsonl(args.bank_manifest))
    records = records[:args.max_samples] if args.max_samples is not None else records
    feature_paths = {}
    for name, checkpoint in checkpoint_items(config):
        path = run_dir / "features" / name / f"{args.feature_set}.pt"
        extract_checkpoint(name, checkpoint, records, config, path, args.zero_actions)
        feature_paths[name] = path
    rows = drift_rows(feature_paths, run_dir.name, config)
    output = run_dir / "metrics" / f"feature_drift_{args.feature_set}.csv"
    write_csv(output, rows)
    print(f"Wrote features and drift metrics: {output}")


if __name__ == "__main__":
    main()

