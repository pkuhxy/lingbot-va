"""Generate the mandatory stage-0 plots from long-form metric tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from common.io import read_csv


def numeric_rows(path, metric):
    output = []
    for row in read_csv(path):
        if row.get("metric_name") != metric:
            continue
        try:
            row["value"] = float(row["metric_value"])
            row["layer_value"] = int(row["layer"]) if row.get("layer") else None
            output.append(row)
        except ValueError:
            pass
    return output


def line_plot(rows, x_key, output, xlabel, ylabel, group_keys=("checkpoint",)):
    import matplotlib.pyplot as plt

    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        x = row[x_key]
        if x is None or x == "":
            continue
        key = tuple(row.get(name, "") for name in group_keys)
        grouped[key][x].append(row["value"])
    if not grouped:
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    for key, points in sorted(grouped.items()):
        xs = sorted(points)
        ys = [sum(points[x]) / len(points[x]) for x in xs]
        axis.plot(xs, ys, marker="o", label="/".join(str(part) for part in key))
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def categorical_plot(rows, category, output, ylabel):
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("checkpoint", ""), row.get(category, ""))].append(row["value"])
    if not grouped:
        return
    labels, values = [], []
    for key, samples in sorted(grouped.items()):
        labels.append("/".join(key))
        values.append(sum(samples) / len(samples))
    figure, axis = plt.subplots(figsize=(max(9, len(labels) * 0.45), 5))
    axis.bar(range(len(labels)), values)
    axis.set_xticks(range(len(labels)), labels, rotation=70, ha="right")
    axis.set_ylabel(ylabel)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    metrics, plots = run_dir / "metrics", run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    feature = metrics / "feature_drift_clean.csv"
    if feature.is_file():
        line_plot(numeric_rows(feature, "linear_cka"), "layer_value", plots / "cka_vs_layer.png", "layer", "linear CKA")
        line_plot(numeric_rows(feature, "cosine_drift_mean"), "layer_value", plots / "cosine_drift_vs_layer.png", "layer", "cosine drift")
        categorical_plot(numeric_rows(feature, "task_linear_cka"), "task", plots / "task_feature_distribution.png", "task CKA")
    action = metrics / "action_sensitivity_train.csv"
    if action.is_file():
        categorical_plot(numeric_rows(action, "velocity_relative_l2"), "perturbation", plots / "action_sensitivity_by_perturbation.png", "velocity relative L2")
    domain = metrics / "domain_gap.csv"
    if domain.is_file():
        line_plot(numeric_rows(domain, "knn_k1_mean"), "layer_value", plots / "domain_knn_vs_layer.png", "layer", "clean-C2R kNN", ("checkpoint", "c2r_split"))
        categorical_plot(numeric_rows(domain, "delta_vs_base/knn_k1"), "c2r_split", plots / "domain_delta_by_split.png", "delta kNN vs base")
    print(f"Plots written to {plots}")


if __name__ == "__main__":
    main()
