#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATA="${DATA:-/mnt/data/share/data/robbyant/robotwin-clean-and-aug-lerobot/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50}"
MANIFEST="output/robotwin_stage0/manifests/clean_full_50.jsonl"
if [[ ! -s "${MANIFEST}" ]]; then
  python evaluation/robotwin/stage0/build_clean_probe_manifest_fixed.py \
    --clean-dataset-path "${DATA}" --output "${MANIFEST}" \
    --episodes-per-task 1 --windows-per-episode 1 --split clean_train_probe
fi

exec bash "${SCRIPT_DIR}/run_stage0_full.sh" "$@"
