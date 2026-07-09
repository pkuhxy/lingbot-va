#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

CASE_NAME="robotwin_clean_step5000"
CKPT_PATH="/mnt/data/users/xianyi/EmbodyAi/lingbot-va/ckpts/train_out_robotwin_clean/checkpoints/checkpoint_step_5000"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/c2r_bench_result}"
SAVE_ROOT="${RESULT_ROOT}/${CASE_NAME}"

export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${PROJECT_ROOT}/RoboTwin}"
export PRETRAINED_MODEL="${CKPT_PATH}"
export AUTO_START_SERVER="${AUTO_START_SERVER:-True}"
export VALIDATED_MANIFEST_DIR="${VALIDATED_MANIFEST_DIR:-${PROJECT_ROOT}/evaluation/robotwin/rt_c2r_validated_manifests}"
export MANIFEST_DIR="${MANIFEST_DIR:-${VALIDATED_MANIFEST_DIR}}"
export STRICT_SEED_MANIFEST="${STRICT_SEED_MANIFEST:-True}"
export ALLOW_MANIFEST_EXPERT_FALLBACK="${ALLOW_MANIFEST_EXPERT_FALLBACK:-True}"
export SAVE_VISUALIZATION="${SAVE_VISUALIZATION:-False}"
export NUM_GPUS="${NUM_GPUS:-1}"
export SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
export START_PORT="${START_PORT:-29056}"
export MASTER_PORT="${MASTER_PORT:-29061}"
export SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
export LOG_DIR="${SAVE_ROOT}/client_logs"
export PID_DIR="${SAVE_ROOT}/pids"
export SERVER_LOG_DIR="${SAVE_ROOT}/server_logs"
export SERVER_SAVE_ROOT="${SAVE_ROOT}/server_visualization"

SEED="${SEED:-0}"
TEST_NUM="${TEST_NUM:-3}"

mkdir -p "${SAVE_ROOT}"

echo "case=${CASE_NAME}"
echo "ckpt=${CKPT_PATH}"
echo "save_root=${SAVE_ROOT}"
echo "manifest_dir=${MANIFEST_DIR}"
echo "test_num=${TEST_NUM}"

bash "${SCRIPT_DIR}/launch_rt_c2r_benchmark.sh" "${SAVE_ROOT}" "${SEED}" "${TEST_NUM}"
