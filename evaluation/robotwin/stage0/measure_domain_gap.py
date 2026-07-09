"""Measure clean-to-C2R feature gap with kNN, MMD, and group-split linear probes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.config import checkpoint_items, load_config
from common.io import get_run_dir, metric_row, write_csv
from common.metrics import domain_logistic_metrics, median_bandwidth, rbf_mmd2


def load_feature(run_dir, checkpoint, name):
    path = run_dir / "features" / checkpoint / f"{name}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature set: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def normalized_window(feature, layer):
    return F.normalize(feature["layers"][str(layer)]["condition_video"]["window"].float(), dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--clean-feature-set", default="clean_domain")
    parser.add_argument("--c2r-feature-set", default="c2r")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = get_run_dir(config, args.run_dir)
    names = [name for name, _ in checkpoint_items(config)]
    clean = {name: load_feature(run_dir, name, args.clean_feature_set) for name in names}
    c2r = {name: load_feature(run_dir, name, args.c2r_feature_set) for name in names}
    layers = sorted(int(layer) for layer in clean["base"]["layers"])
    bandwidths = {layer: median_bandwidth(normalized_window(clean["base"], layer)) for layer in layers}
    rows, aggregates = [], {}
    for checkpoint in names:
        clean_meta, c2r_meta = clean[checkpoint]["metadata"], c2r[checkpoint]["metadata"]
        splits = sorted({row["c2r_split"] for row in c2r_meta})
        for layer in layers:
            clean_x, c2r_x = normalized_window(clean[checkpoint], layer), normalized_window(c2r[checkpoint], layer)
            for split in splits:
                shifted_idx = [i for i, row in enumerate(c2r_meta) if row["c2r_split"] == split]
                per_k = defaultdict(list)
                for index in shifted_idx:
                    meta = c2r_meta[index]
                    candidates = [i for i, row in enumerate(clean_meta) if row["task"] == meta["task"]]
                    if not candidates:
                        continue
                    distances = torch.cdist(c2r_x[index:index+1], clean_x[candidates]).squeeze(0)
                    for k in (1, 5):
                        if len(candidates) >= k:
                            value = float(distances.topk(k, largest=False).values.mean())
                            per_k[k].append(value)
                            rows.append(metric_row(
                                run_id=run_dir.name, checkpoint=checkpoint, sample_id=meta["sample_id"],
                                task=meta["task"], episode_or_seed=meta["episode_or_seed"], domain="c2r",
                                c2r_split=split, progress_bin=meta["progress_bin"], layer=layer,
                                token_group="condition_video", metric_name=f"knn_k{k}", metric_value=value,
                            ))
                for k, values in per_k.items():
                    mean = float(np.mean(values))
                    aggregates[(checkpoint, split, layer, f"knn_k{k}")] = mean
                    rows.append(metric_row(
                        run_id=run_dir.name, checkpoint=checkpoint, sample_id="aggregate", task="all",
                        episode_or_seed="all", domain="c2r", c2r_split=split, layer=layer,
                        token_group="condition_video", metric_name=f"knn_k{k}_mean", metric_value=mean,
                    ))
                task_mmd = []
                for task in sorted({c2r_meta[i]["task"] for i in shifted_idx}):
                    ci = [i for i, row in enumerate(clean_meta) if row["task"] == task]
                    si = [i for i in shifted_idx if c2r_meta[i]["task"] == task]
                    if len(ci) >= 2 and len(si) >= 2:
                        task_mmd.append(rbf_mmd2(clean_x[ci], c2r_x[si], bandwidths[layer]))
                if task_mmd:
                    value = float(np.mean(task_mmd))
                    aggregates[(checkpoint, split, layer, "mmd_rbf")] = value
                    rows.append(metric_row(
                        run_id=run_dir.name, checkpoint=checkpoint, sample_id="aggregate", task="all",
                        episode_or_seed="all", domain="c2r", c2r_split=split, layer=layer,
                        token_group="condition_video", metric_name="mmd_rbf", metric_value=value,
                        bandwidth=bandwidths[layer],
                    ))
                n = min(len(clean_x), len(shifted_idx))
                if n >= 4:
                    features = torch.cat([clean_x[:n], c2r_x[shifted_idx[:n]]]).numpy()
                    labels = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
                    groups = [f"clean:{row['episode_or_seed']}" for row in clean_meta[:n]] + [f"c2r:{c2r_meta[i]['episode_or_seed']}" for i in shifted_idx[:n]]
                    try:
                        probe = domain_logistic_metrics(features, labels, groups, seed=config["runtime"]["seed"], c=1.0)
                    except ValueError:
                        probe = {}
                    for metric, value in probe.items():
                        aggregates[(checkpoint, split, layer, metric)] = value
                        rows.append(metric_row(
                            run_id=run_dir.name, checkpoint=checkpoint, sample_id="aggregate", task="all",
                            episode_or_seed="group_split", domain="c2r", c2r_split=split, layer=layer,
                            token_group="condition_video", metric_name=f"domain_{metric}", metric_value=value,
                        ))
    for (checkpoint, split, layer, metric), value in list(aggregates.items()):
        base = aggregates.get(("base", split, layer, metric))
        if base is not None:
            rows.append(metric_row(
                run_id=run_dir.name, checkpoint=checkpoint, sample_id="aggregate", task="all",
                episode_or_seed="all", domain="c2r", c2r_split=split, layer=layer,
                token_group="condition_video", metric_name=f"delta_vs_base/{metric}",
                metric_value=value - base,
            ))
    output = run_dir / "metrics" / "domain_gap.csv"
    write_csv(output, rows)
    print(f"Wrote {len(rows)} metrics: {output}")


if __name__ == "__main__":
    main()

