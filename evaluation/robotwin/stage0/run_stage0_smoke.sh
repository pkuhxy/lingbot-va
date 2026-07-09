#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

CONFIG="${CONFIG:-evaluation/robotwin/stage0/configs/robotwin_stage0.yaml}"
DATA="${DATA:-/mnt/data/share/data/robbyant/robotwin-clean-and-aug-lerobot/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50}"
RUN_ID="${RUN_ID:-stage0_001}"
GPU_ID="${GPU_ID:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
POLICY_MAX_SAMPLES="${POLICY_MAX_SAMPLES:-1}"
RUN="output/robotwin_stage0/${RUN_ID}"
MANIFEST="output/robotwin_stage0/manifests/clean_smoke.jsonl"
BANK="output/robotwin_stage0/banks/clean_smoke"
AUDIT="${RUN}/checkpoint_audit/checkpoint_audit.json"

mkdir -p output/robotwin_stage0/manifests output/robotwin_stage0/banks
exec > >(tee -a "output/robotwin_stage0/launcher_${RUN_ID}.log") 2>&1
trap 'code=$?; echo "FAILED at line ${BASH_LINENO[0]} (exit=${code})" >&2; exit ${code}' ERR

[[ -s "${AUDIT}" ]] || { echo "Missing passed audit: ${AUDIT}" >&2; exit 2; }
python evaluation/robotwin/stage0/validate_checkpoint_audit.py --audit "${AUDIT}" --in-place

if [[ ! -s "${MANIFEST}" ]]; then
  python evaluation/robotwin/stage0/build_clean_probe_manifest.py \
    --clean-dataset-path "${DATA}" --output "${MANIFEST}" \
    --episodes-per-task 1 --windows-per-episode 1 --split clean_train_probe
fi

if [[ ! -s "${BANK}/manifest.jsonl" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python evaluation/robotwin/stage0/build_paired_latent_bank.py \
    --config "${CONFIG}" --manifest "${MANIFEST}" --output-dir "${BANK}" \
    --max-samples "${MAX_SAMPLES}" --skip-directional
fi

if [[ ! -s "${RUN}/metrics/feature_drift_clean.csv" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python evaluation/robotwin/stage0/extract_train_features.py \
    --config "${CONFIG}" --bank-manifest "${BANK}/manifest.jsonl" \
    --run-dir "${RUN}" --feature-set clean --max-samples "${MAX_SAMPLES}"
fi

if [[ ! -s "${RUN}/metrics/action_sensitivity_train.csv" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python evaluation/robotwin/stage0/measure_action_sensitivity.py \
    --config "${CONFIG}" --bank-manifest "${BANK}/manifest.jsonl" \
    --run-dir "${RUN}" --mode train --max-samples "${MAX_SAMPLES}"
fi

if [[ ! -s "${RUN}/metrics/action_sensitivity_policy.csv" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python evaluation/robotwin/stage0/measure_action_sensitivity.py \
    --config "${CONFIG}" --bank-manifest "${BANK}/manifest.jsonl" \
    --run-dir "${RUN}" --mode policy --max-samples "${POLICY_MAX_SAMPLES}"
fi

if [[ ! -s "${RUN}/metrics/video_dynamics.csv" ]]; then
  python evaluation/robotwin/stage0/measure_video_dynamics.py --run-dir "${RUN}"
fi

python evaluation/robotwin/stage0/plot_stage0_metrics.py --run-dir "${RUN}"
python evaluation/robotwin/stage0/aggregate_stage0_report.py --run-dir "${RUN}"
echo "DONE: ${RUN}/report.md"
