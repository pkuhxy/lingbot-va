"""Revalidate a stage-0 audit with narrowly scoped base-checkpoint compatibility rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.io import write_json


ALLOWED_BASE_ONLY_PARAMETERS = {
    # The upstream base safetensors retain a legacy Conv3d patch embed. The
    # current WanTransformer3DModel defines and calls patch_embedding_mlp only,
    # so full-state training checkpoints correctly omit these inactive tensors.
    "patch_embedding.bias",
    "patch_embedding.weight",
}


def validate(result: dict) -> dict:
    failures = []
    for name, comparison in result["comparisons"].items():
        missing = set(comparison.get("missing_parameters", []))
        ignored = sorted(missing & ALLOWED_BASE_ONLY_PARAMETERS)
        remaining = sorted(missing - ALLOWED_BASE_ONLY_PARAMETERS)
        comparison["ignored_base_only_parameters"] = ignored
        comparison["missing_parameters"] = remaining
        if remaining or comparison.get("unexpected_parameters") or comparison.get("shape_mismatch"):
            failures.append(f"{name}: parameter name/shape mismatch")
        overall = comparison.get("groups", {}).get("__all__", {})
        if int(overall.get("nonfinite", 0)):
            failures.append(f"{name}: non-finite parameters")
        relative = float(overall.get("relative_l2", 0.0))
        if name == "base" and relative != 0.0:
            failures.append("base-vs-base relative L2 is non-zero")
        if name != "base" and relative == 0.0:
            failures.append(f"{name}: transformer has no update")
    result["compatibility_policy"] = {
        "allowed_base_only_parameters": sorted(ALLOWED_BASE_ONLY_PARAMETERS),
        "reason": "inactive legacy patch_embedding; current model uses patch_embedding_mlp",
    }
    result["failures"] = failures
    result["passed"] = not failures
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--in-place", action="store_true",
        help="Replace the input JSON after revalidation; otherwise write checkpoint_audit_validated.json.",
    )
    args = parser.parse_args()
    audit_path = args.audit.expanduser().resolve()
    with audit_path.open("r", encoding="utf-8") as handle:
        result = validate(json.load(handle))
    output = audit_path if args.in_place else audit_path.with_name("checkpoint_audit_validated.json")
    write_json(output, result)
    print(f"Checkpoint audit {'PASSED' if result['passed'] else 'FAILED'}: {output}")
    if result["failures"]:
        for failure in result["failures"]:
            print(f"  - {failure}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
