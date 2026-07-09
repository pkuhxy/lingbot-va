#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_ID="${RUN_ID:-stage0_full_001}"
POLICY_SAMPLES="${POLICY_SAMPLES:-10}"
GPU_IDS="${GPU_IDS:-0,1}"
CONTINUE_FULL="${CONTINUE_FULL:-1}"
CONFIG="${CONFIG:-evaluation/robotwin/stage0/configs/robotwin_stage0.yaml}"
BANK="${BANK:-output/robotwin_stage0/banks/clean_full_50/manifest.jsonl}"
RUN="output/robotwin_stage0/${RUN_ID}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
case "${POLICY_SAMPLES}" in ''|*[!0-9]*) echo "POLICY_SAMPLES must be a positive integer" >&2; exit 2;; esac
(( POLICY_SAMPLES > 0 )) || { echo "POLICY_SAMPLES must be positive" >&2; exit 2; }
(( POLICY_SAMPLES >= NUM_GPUS )) || { echo "POLICY_SAMPLES must be >= number of GPUs (${NUM_GPUS})" >&2; exit 2; }
[[ -d "${RUN}" ]] || { echo "Missing run directory: ${RUN}" >&2; exit 2; }
[[ -s "${BANK}" ]] || { echo "Missing latent bank: ${BANK}" >&2; exit 2; }

if command -v pgrep >/dev/null 2>&1 && timeout 5s pgrep -f '[m]easure_action_sensitivity_streaming.py.*--mode policy' >/dev/null 2>&1; then
  echo "Another policy action-sensitivity process is still running." >&2
  echo "Stop the previous 50-sample launcher first (Ctrl-C), then rerun this script." >&2
  exit 3
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PARALLEL_DIR="${RUN}/policy_parallel_${POLICY_SAMPLES}_${STAMP}"
MANIFEST_DIR="${PARALLEL_DIR}/manifests"
COMBINED_RUN="${PARALLEL_DIR}/combined"
LOG="${PARALLEL_DIR}/launcher.log"
mkdir -p "${MANIFEST_DIR}" "${COMBINED_RUN}/metrics" "${COMBINED_RUN}/paired_predictions" "${COMBINED_RUN}/plots"
exec > >(tee -a "${LOG}") 2>&1

BANK_RECORDS="$(wc -l < "${BANK}")"
(( POLICY_SAMPLES <= BANK_RECORDS )) || { echo "POLICY_SAMPLES=${POLICY_SAMPLES} exceeds bank size ${BANK_RECORDS}" >&2; exit 2; }
echo "Stage-0 policy parallel run=${RUN_ID} samples=${POLICY_SAMPLES} GPUs=${GPU_IDS}"
awk -v total="${BANK_RECORDS}" -v want="${POLICY_SAMPLES}" -v shards="${NUM_GPUS}" -v out="${MANIFEST_DIR}" 'BEGIN { if (want == 1) { selected[1] = 0 } else { for (i = 0; i < want; i++) { line = int(i * (total - 1) / (want - 1) + 0.5) + 1; selected[line] = i } } } (NR in selected) { shard = selected[NR] % shards; file = sprintf("%s/shard_%d.jsonl", out, shard); print $0 >> file; close(file) }' "${BANK}"

declare -a pids=()
declare -a shard_runs=()
heartbeat_pid=""
cleanup() {
  local pid
  [[ -z "${heartbeat_pid}" ]] || kill "${heartbeat_pid}" 2>/dev/null || true
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
}
trap cleanup INT TERM

for shard_index in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[${shard_index}]}"
  shard_manifest="${MANIFEST_DIR}/shard_${shard_index}.jsonl"
  shard_run="${PARALLEL_DIR}/gpu_${gpu}"
  shard_log="${shard_run}/policy.log"
  [[ -s "${shard_manifest}" ]] || { echo "Empty shard manifest: ${shard_manifest}" >&2; exit 2; }
  mkdir -p "${shard_run}/metrics" "${shard_run}/paired_predictions" "${shard_run}/plots"
  shard_runs+=("${shard_run}")
  echo "Launching shard=${shard_index} samples=$(wc -l < "${shard_manifest}") GPU=${gpu} log=${shard_log}"
  (
    env CUDA_VISIBLE_DEVICES="${gpu}" python -u "${SCRIPT_DIR}/measure_action_sensitivity_streaming.py" --config "${CONFIG}" --bank-manifest "${shard_manifest}" --run-dir "${shard_run}" --mode policy
  ) >"${shard_log}" 2>&1 &
  pids+=("$!")
done

heartbeat() {
  while true; do
    sleep 60
    echo "[$(date -u +%FT%TZ)] policy shards still running; logs: ${PARALLEL_DIR}/gpu_*/policy.log"
  done
}
heartbeat &
heartbeat_pid="$!"

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    echo "Completed GPU=${GPU_ARRAY[${index}]}"
  else
    status="$?"
    echo "FAILED GPU=${GPU_ARRAY[${index}]} exit=${status}; see ${shard_runs[${index}]}/policy.log" >&2
    failed=1
  fi
done
kill "${heartbeat_pid}" 2>/dev/null || true
wait "${heartbeat_pid}" 2>/dev/null || true
heartbeat_pid=""
trap - INT TERM
(( failed == 0 )) || exit 1

TARGET_METRIC="${RUN}/metrics/action_sensitivity_policy.csv"
if [[ -s "${TARGET_METRIC}" ]]; then cp "${TARGET_METRIC}" "${PARALLEL_DIR}/previous_action_sensitivity_policy.csv"; fi
cp "${shard_runs[0]}/metrics/action_sensitivity_policy.csv" "${TARGET_METRIC}"
for (( index = 1; index < NUM_GPUS; index++ )); do
  tail -n +2 "${shard_runs[${index}]}/metrics/action_sensitivity_policy.csv" >> "${TARGET_METRIC}"
done
cp "${TARGET_METRIC}" "${COMBINED_RUN}/metrics/action_sensitivity_policy.csv"
echo "Merged $(( $(wc -l < "${TARGET_METRIC}") - 1 )) policy metric rows: ${TARGET_METRIC}"

for shard_run in "${shard_runs[@]}"; do
  while IFS= read -r source; do
    relative="${source#${shard_run}/paired_predictions/}"
    target="${COMBINED_RUN}/paired_predictions/${relative}"
    mkdir -p "$(dirname "${target}")"
    ln "${source}" "${target}"
  done < <(find "${shard_run}/paired_predictions" -type f -name '*.pt' -print)
done

python "${SCRIPT_DIR}/measure_video_dynamics.py" --run-dir "${COMBINED_RUN}"
if [[ -s "${RUN}/metrics/video_dynamics.csv" ]]; then cp "${RUN}/metrics/video_dynamics.csv" "${PARALLEL_DIR}/previous_video_dynamics.csv"; fi
cp "${COMBINED_RUN}/metrics/video_dynamics.csv" "${RUN}/metrics/video_dynamics.csv"
cp "${COMBINED_RUN}/metrics/video_whitening.pt" "${RUN}/metrics/video_whitening.pt"
if [[ -s "${COMBINED_RUN}/plots/video_dynamics_vs_action.png" ]]; then cp "${COMBINED_RUN}/plots/video_dynamics_vs_action.png" "${RUN}/plots/video_dynamics_vs_action.png"; fi
echo "Merged policy/video metrics into ${RUN}/metrics"

if [[ "${CONTINUE_FULL}" == "1" ]]; then
  echo "Continuing the remaining Stage-0 stages..."
  RUN_ID="${RUN_ID}" POLICY_SAMPLES="${POLICY_SAMPLES}" GPU_ID="${GPU_ARRAY[0]}" bash "${SCRIPT_DIR}/run_stage0_full_fixed.sh"
else
  python "${SCRIPT_DIR}/plot_stage0_metrics.py" --run-dir "${RUN}"
  python "${SCRIPT_DIR}/aggregate_stage0_report.py" --run-dir "${RUN}"
fi

echo "POLICY PARALLEL COMPLETE: ${RUN}/report.md"
