#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:-stage0_full_001}"
GPU_ID="${GPU_ID:-0}"
POLICY_SAMPLES="${POLICY_SAMPLES:-50}"
DIRECTIONAL_SAMPLES="${DIRECTIONAL_SAMPLES:-10}"
C2R_PER_SPLIT="${C2R_PER_SPLIT:-50}"
RUN_CLEAN_ROLLOUT="${RUN_CLEAN_ROLLOUT:-1}"
CONFIG="${CONFIG:-evaluation/robotwin/stage0/configs/robotwin_stage0.yaml}"
DATA="${DATA:-/mnt/data/share/data/robbyant/robotwin-clean-and-aug-lerobot/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${ROOT}/RoboTwin}"
RUN="output/robotwin_stage0/${RUN_ID}"
MANIFEST="output/robotwin_stage0/manifests/clean_full_50.jsonl"
BANK_DIR="output/robotwin_stage0/banks/clean_full_50"
BANK="${BANK_DIR}/manifest.jsonl"
DIR_BANK_DIR="output/robotwin_stage0/banks/clean_directional_${DIRECTIONAL_SAMPLES}"
DIR_BANK="${DIR_BANK_DIR}/manifest.jsonl"
PROMPT_MANIFEST_DIR="output/robotwin_stage0/manifests/c2r_prompt_matched"
C2R_BANK_DIR="output/robotwin_stage0/banks/c2r_initial_4x${C2R_PER_SPLIT}"
C2R_BANK="${C2R_BANK_DIR}/manifest.jsonl"
LOG="output/robotwin_stage0/launcher_${RUN_ID}.log"

export PYTHONPATH="${ROOT}:${ROBOTWIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p output/robotwin_stage0
exec > >(tee -a "${LOG}") 2>&1
trap 'code=$?; echo "FAILED line=${BASH_LINENO[0]} exit=${code} log=${LOG}" >&2; exit ${code}' ERR

run() { echo; printf '>>>'; printf ' %q' "$@"; echo; "$@"; }
skip_or_run() {
  local output=$1; shift
  if [[ -s "${output}" ]]; then echo "SKIP existing: ${output}"; else run "$@"; fi
}

echo "Stage-0 full run=${RUN_ID} gpu=${GPU_ID} policy=${POLICY_SAMPLES} c2r=4x${C2R_PER_SPLIT}"
[[ -d "${DATA}" ]] || { echo "Missing clean data: ${DATA}" >&2; exit 2; }
[[ -d "${ROBOTWIN_ROOT}/envs" ]] || { echo "Missing RoboTwin: ${ROBOTWIN_ROOT}" >&2; exit 2; }

# M0
AUDIT="${RUN}/checkpoint_audit/checkpoint_audit.json"
if [[ -s "${AUDIT}" ]]; then
  run python evaluation/robotwin/stage0/validate_checkpoint_audit.py --audit "${AUDIT}" --in-place
elif [[ -d "${RUN}" ]]; then
  run python evaluation/robotwin/stage0/audit_checkpoints_compatible.py --config "${CONFIG}" --run-dir "${RUN}"
else
  run python evaluation/robotwin/stage0/audit_checkpoints_compatible.py --config "${CONFIG}" --run-id "${RUN_ID}"
fi

# Fifty task-balanced clean observations.
skip_or_run "${MANIFEST}" \
  python evaluation/robotwin/stage0/build_clean_probe_manifest.py \
    --clean-dataset-path "${DATA}" --output "${MANIFEST}" \
    --episodes-per-task 1 --windows-per-episode 1 --split clean_train_probe

skip_or_run "${BANK}" \
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python evaluation/robotwin/stage0/build_paired_latent_bank_streaming.py \
    --config "${CONFIG}" --manifest "${MANIFEST}" --output-dir "${BANK_DIR}" \
    --max-samples 50 --skip-directional

# M1
skip_or_run "${RUN}/metrics/feature_drift_clean.csv" \
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python evaluation/robotwin/stage0/extract_train_features.py \
    --config "${CONFIG}" --bank-manifest "${BANK}" --run-dir "${RUN}" \
    --feature-set clean --max-samples 50

# M2 train-forward and true policy path.
skip_or_run "${RUN}/metrics/action_sensitivity_train.csv" \
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python evaluation/robotwin/stage0/measure_action_sensitivity.py \
    --config "${CONFIG}" --bank-manifest "${BANK}" --run-dir "${RUN}" \
    --mode train --max-samples 50

skip_or_run "${RUN}/metrics/action_sensitivity_policy.csv" \
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python evaluation/robotwin/stage0/measure_action_sensitivity_streaming.py \
    --config "${CONFIG}" --bank-manifest "${BANK}" --run-dir "${RUN}" \
    --mode policy --max-samples "${POLICY_SAMPLES}"

skip_or_run "${RUN}/metrics/video_dynamics.csv" \
  python evaluation/robotwin/stage0/measure_video_dynamics.py --run-dir "${RUN}"

# M4 directional finite difference on a bounded subset.
skip_or_run "${DIR_BANK}" \
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python evaluation/robotwin/stage0/build_paired_latent_bank_streaming.py \
    --config "${CONFIG}" --manifest "${MANIFEST}" --output-dir "${DIR_BANK_DIR}" \
    --max-samples "${DIRECTIONAL_SAMPLES}"

skip_or_run "${RUN}/metrics/directional_fd.csv" \
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python evaluation/robotwin/stage0/measure_directional_jacobian.py \
    --config "${CONFIG}" --bank-manifest "${DIR_BANK}" --run-dir "${RUN}" \
    --max-samples "${DIRECTIONAL_SAMPLES}"

# M3 task/prompt-matched initial C2R observations.
PROMPT_SENTINEL="${PROMPT_MANIFEST_DIR}/rt_c2r_height.jsonl"
skip_or_run "${PROMPT_SENTINEL}" \
  python evaluation/robotwin/stage0/attach_c2r_prompts.py \
    --clean-manifest "${MANIFEST}" \
    --validated-manifest evaluation/robotwin/rt_c2r_validated_manifests/rt_c2r_background.jsonl \
    --validated-manifest evaluation/robotwin/rt_c2r_validated_manifests/rt_c2r_light.jsonl \
    --validated-manifest evaluation/robotwin/rt_c2r_validated_manifests/rt_c2r_clutter.jsonl \
    --validated-manifest evaluation/robotwin/rt_c2r_validated_manifests/rt_c2r_height.jsonl \
    --output-dir "${PROMPT_MANIFEST_DIR}"

skip_or_run "${C2R_BANK}" \
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python evaluation/robotwin/stage0/collect_c2r_initial_bank.py \
    --config "${CONFIG}" --robotwin-root "${ROBOTWIN_ROOT}" \
    --validated-manifest "${PROMPT_MANIFEST_DIR}/rt_c2r_background.jsonl" \
    --validated-manifest "${PROMPT_MANIFEST_DIR}/rt_c2r_light.jsonl" \
    --validated-manifest "${PROMPT_MANIFEST_DIR}/rt_c2r_clutter.jsonl" \
    --validated-manifest "${PROMPT_MANIFEST_DIR}/rt_c2r_height.jsonl" \
    --output-dir "${C2R_BANK_DIR}" --max-per-split "${C2R_PER_SPLIT}"

skip_or_run "${RUN}/features/base/clean_domain.pt" \
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python evaluation/robotwin/stage0/extract_train_features.py \
    --config "${CONFIG}" --bank-manifest "${BANK}" --run-dir "${RUN}" \
    --feature-set clean_domain --zero-actions --max-samples 50

skip_or_run "${RUN}/features/base/c2r.pt" \
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python evaluation/robotwin/stage0/extract_train_features.py \
    --config "${CONFIG}" --bank-manifest "${C2R_BANK}" --run-dir "${RUN}" \
    --feature-set c2r --zero-actions

skip_or_run "${RUN}/metrics/domain_gap.csv" \
  python evaluation/robotwin/stage0/measure_domain_gap.py \
    --config "${CONFIG}" --run-dir "${RUN}"

run python evaluation/robotwin/stage0/plot_stage0_metrics.py --run-dir "${RUN}"
run python evaluation/robotwin/stage0/aggregate_stage0_report.py --run-dir "${RUN}"

# Comparable clean checkpoint rollout coverage: all 50 easy/background tasks.
if [[ "${RUN_CLEAN_ROLLOUT}" == "1" ]]; then
  run env ROBOTWIN_ROOT="${ROBOTWIN_ROOT}" SPLITS="easy background" \
    NUM_GPUS=2 GPU_IDS=0,1 START_PORT=29256 MASTER_PORT=29261 \
    AUTO_START_SERVER=True TEST_NUM=3 \
    bash evaluation/robotwin/run_rt_c2r_robotwin_clean_step5000.sh
fi

echo "FULL STAGE-0 COMPLETE: ${RUN}/report.md"
