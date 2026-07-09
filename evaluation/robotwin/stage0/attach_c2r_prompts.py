"""Attach clean task prompts to validated C2R manifests for matched feature extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.io import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--validated-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    prompts = {}
    for record in read_jsonl(args.clean_manifest):
        prompts.setdefault(record["task"], record["prompt"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in args.validated_manifest:
        rows = []
        for record in read_jsonl(source):
            if record["task"] not in prompts:
                raise KeyError(f"No clean prompt for C2R task {record['task']}")
            rows.append({**record, "prompt": prompts[record["task"]]})
        output = args.output_dir / source.name
        write_jsonl(output, rows)
        print(f"Wrote {len(rows)} rows: {output}")


if __name__ == "__main__":
    main()
