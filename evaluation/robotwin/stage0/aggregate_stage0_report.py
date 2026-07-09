"""Aggregate stage-0 long tables into fixed plots and a diagnostic report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common.io import read_csv


SECTIONS = (
    "Checkpoints and data", "Determinism and sanity checks", "Parameter update audit",
    "Layer-wise feature drift", "Action sensitivity", "Future-video dynamics stability",
    "Directional finite difference", "Clean-C2R domain gap", "Joint diagnosis", "Limitations",
)


def summarize(path):
    rows = read_csv(path)
    by_metric = defaultdict(list)
    for row in rows:
        try:
            by_metric[(row.get("checkpoint", ""), row.get("metric_name", ""))].append(float(row["metric_value"]))
        except (KeyError, ValueError):
            pass
    return rows, {key: sum(values) / len(values) for key, values in by_metric.items() if values}


def table(summary, contains):
    selected = [(key, value) for key, value in summary.items() if contains in key[1]]
    if not selected:
        return "No completed metric table was found."
    lines = ["| checkpoint | metric | mean |", "|---|---|---:|"]
    lines.extend(f"| {checkpoint} | {metric} | {value:.6g} |" for (checkpoint, metric), value in sorted(selected))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    summaries = {}
    for path in sorted((run_dir / "metrics").glob("*.csv")):
        _rows, summaries[path.name] = summarize(path)
    audit_path = run_dir / "checkpoint_audit" / "checkpoint_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else None
    report = ["# Stage 0 Diagnostic Report", ""]
    report += ["## Checkpoints and data", "", f"Run: `{run_dir.name}`", ""]
    report += ["## Determinism and sanity checks", ""]
    report += [f"Checkpoint audit: **{'PASS' if audit and audit.get('passed') else 'NOT PASSED'}**.", ""]
    report += ["Identity rows are retained in action/video tables and must be checked before interpreting perturbations.", ""]
    report += ["## Parameter update audit", ""]
    if audit:
        for name, comparison in audit["comparisons"].items():
            overall = comparison["groups"].get("__all__", {})
            report.append(f"- {name}: relative L2 `{overall.get('relative_l2', 'n/a')}`, nonfinite `{overall.get('nonfinite', 'n/a')}`")
    else:
        report.append("Checkpoint audit has not been run.")
    report += ["", "## Layer-wise feature drift", "", table(summaries.get("feature_drift_clean.csv", {}), "cka"), ""]
    report += ["## Action sensitivity", "", table(summaries.get("action_sensitivity_policy.csv", summaries.get("action_sensitivity_train.csv", {})), "relative_l2"), ""]
    report += ["## Future-video dynamics stability", "", table(summaries.get("video_dynamics.csv", {}), "latent_residual"), ""]
    report += ["## Directional finite difference", "", table(summaries.get("directional_fd.csv", {}), "directional"), ""]
    report += ["## Clean-C2R domain gap", "", table(summaries.get("domain_gap.csv", {}), "delta_vs_base"), ""]
    report += [
        "## Joint diagnosis", "",
        "Apply the fixed decision table from `STAGE0_TEST_IMPLEMENTATION_GUIDE.md`; do not infer policy quality from CKA or a single offline metric.", "",
        "## Limitations", "",
        "Latent residual dynamics remain VAE-dependent. Initial and final C2R observations are reported separately. Closed-loop rollout success remains the behavioral ground truth.", "",
    ]
    output = run_dir / "report.md"
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote report: {output}")


if __name__ == "__main__":
    main()

