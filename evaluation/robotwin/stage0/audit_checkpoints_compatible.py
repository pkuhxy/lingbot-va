"""Audit checkpoints with documented upstream-base legacy-key compatibility."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.checkpoint import (
    SafeTensorStore, align_head_audit, audit_pair, checkpoint_metadata,
    plot_parameter_drift,
)
from common.config import checkpoint_items, load_config
from common.io import get_run_dir, write_json
from validate_checkpoint_audit import validate


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
    result = {"checkpoints": {}, "comparisons": {}, "passed": False, "failures": []}
    for name, path in checkpoints:
        result["checkpoints"][name] = checkpoint_metadata(path)
        result["checkpoints"][name]["align_head"] = align_head_audit(path)
        result["comparisons"][name] = audit_pair(base, SafeTensorStore(path))
    result = validate(result)
    output = run_dir / "checkpoint_audit" / "checkpoint_audit.json"
    write_json(output, result)
    if not args.skip_plot:
        plot_parameter_drift(result, run_dir / "plots" / "parameter_drift_by_layer.png")
    print(f"Checkpoint audit {'PASSED' if result['passed'] else 'FAILED'}: {output}")
    if result["failures"]:
        for failure in result["failures"]:
            print(f"  - {failure}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
