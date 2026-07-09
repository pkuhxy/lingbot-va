#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-evaluation/robotwin/stage0/configs/robotwin_stage0.yaml}"
DATA="${DATA:-/mnt/data/share/data/robbyant/robotwin-clean-and-aug-lerobot/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50}"
GPU_ID="${GPU_ID:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
MANIFEST="output/robotwin_stage0/manifests/clean_smoke.jsonl"
BANK="output/robotwin_stage0/banks/clean_smoke"

if [[ ! -s "${MANIFEST}" ]]; then
  python evaluation/robotwin/stage0/build_clean_probe_manifest.py \
    --clean-dataset-path "${DATA}" --output "${MANIFEST}" \
    --episodes-per-task 1 --windows-per-episode 1 --split clean_train_probe
fi

if [[ ! -s "${BANK}/manifest.jsonl" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python evaluation/robotwin/stage0/build_paired_latent_bank_aligned.py \
    --config "${CONFIG}" --manifest "${MANIFEST}" --output-dir "${BANK}" \
    --max-samples "${MAX_SAMPLES}" --skip-directional
fi

exec bash "${SCRIPT_DIR}/run_stage0.sh" "$@"
