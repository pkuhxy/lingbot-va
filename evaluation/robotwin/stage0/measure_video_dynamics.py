"""Measure whitened future-video temporal-residual stability from policy predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.io import get_run_dir, metric_row, write_csv
from common.metrics import latent_residual_cosine, relative_l2


def current_frame(result):
    return result["current_latent"][:, :, :1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", default="base")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    base_files = sorted((run_dir / "paired_predictions" / args.base_checkpoint).glob("**/*.pt"))
    if not base_files:
        raise FileNotFoundError("No base policy predictions; run measure_action_sensitivity.py --mode policy first")
    residuals = []
    for path in base_files:
        pair = torch.load(path, map_location="cpu", weights_only=False)
        clean = pair["clean"]
        residuals.append(clean["future_latent"] - current_frame(clean))
    flattened = torch.cat([value.permute(1, 0, 2, 3, 4).reshape(value.shape[1], -1) for value in residuals], dim=1)
    mean = flattened.mean(1).view(1, -1, 1, 1, 1)
    std = flattened.std(1).clamp_min(1e-6).view(1, -1, 1, 1, 1)
    torch.save({"mean": mean, "std": std, "source": [str(path) for path in base_files]}, run_dir / "metrics" / "video_whitening.pt")
    rows, scatter = [], []
    for checkpoint_dir in sorted((run_dir / "paired_predictions").iterdir()):
        if not checkpoint_dir.is_dir():
            continue
        for path in sorted(checkpoint_dir.glob("**/*.pt")):
            pair = torch.load(path, map_location="cpu", weights_only=False)
            clean, augmented = pair["clean"], pair["augmented"]
            record, variant = pair["record"], pair["variant"]
            dyn = latent_residual_cosine(
                clean["future_latent"], augmented["future_latent"],
                current_frame(clean), current_frame(augmented), mean, std,
            )
            action = relative_l2(clean["action"], augmented["action"])
            common = dict(
                run_id=run_dir.name, checkpoint=checkpoint_dir.name,
                sample_id=record["sample_id"], task=record.get("task", ""),
                episode_or_seed=record.get("episode_or_seed", record.get("episode_index", "")),
                domain="clean", perturbation=variant.get("perturbation", "identity"),
                severity=variant.get("severity", 0), transform_seed=variant.get("transform_seed", ""),
            )
            rows.append(metric_row(**common, token_group="future_video", metric_name="latent_residual_cosine", metric_value=dyn))
            rows.append(metric_row(**common, token_group="final_action", metric_name="paired_action_relative_l2", metric_value=action))
            scatter.append((checkpoint_dir.name, dyn, action))
    output = run_dir / "metrics" / "video_dynamics.csv"
    write_csv(output, rows)
    try:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(6, 5))
        for checkpoint in sorted({row[0] for row in scatter}):
            values = [row for row in scatter if row[0] == checkpoint]
            axis.scatter([row[1] for row in values], [row[2] for row in values], s=12, alpha=0.6, label=checkpoint)
        axis.set_xlabel("latent residual cosine distance")
        axis.set_ylabel("action relative L2")
        axis.legend()
        figure.tight_layout()
        figure.savefig(run_dir / "plots" / "video_dynamics_vs_action.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    print(f"Wrote {len(rows)} metrics: {output}")


if __name__ == "__main__":
    main()

