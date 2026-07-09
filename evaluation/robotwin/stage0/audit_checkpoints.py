"""Stream-audit stage-0 transformer checkpoints before GPU diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.checkpoint import (
    SafeTensorStore, align_head_audit, audit_pair, checkpoint_metadata,
    plot_parameter_drift,
)
from common.config import checkpoint_items, load_config
from common.io import get_run_dir, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = get_run_dir(config, args.run_dir, args.run_id)
    checkpoints = checkpoint_items(config)
    base = SafeTensorStore(dict(checkpoints)["base"])
    result = {"checkpoints": {}, "comparisons": {}, "passed": True, "failures": []}
    for name, path in checkpoints:
        result["checkpoints"][name] = checkpoint_metadata(path)
        result["checkpoints"][name]["align_head"] = align_head_audit(path)
        comparison = audit_pair(base, SafeTensorStore(path))
        result["comparisons"][name] = comparison
        if comparison["missing_parameters"] or comparison["unexpected_parameters"] or comparison["shape_mismatch"]:
            result["passed"] = False
            result["failures"].append(f"{name}: parameter name/shape mismatch")
        if comparison["groups"].get("__all__", {}).get("nonfinite", 0):
            result["passed"] = False
            result["failures"].append(f"{name}: non-finite parameters")
        relative = comparison["groups"].get("__all__", {}).get("relative_l2", 0.0)
        if name == "base" and relative != 0.0:
            result["passed"] = False
            result["failures"].append("base-vs-base relative L2 is non-zero")
        if name != "base" and relative == 0.0:
            result["passed"] = False
            result["failures"].append(f"{name}: transformer has no update")
    output = run_dir / "checkpoint_audit" / "checkpoint_audit.json"
    write_json(output, result)
    if not args.skip_plot:
        plot_parameter_drift(result, run_dir / "plots" / "parameter_drift_by_layer.png")
    print(f"Checkpoint audit {'PASSED' if result['passed'] else 'FAILED'}: {output}")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

