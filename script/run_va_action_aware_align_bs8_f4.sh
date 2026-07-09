#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

NGPU="${NGPU:-2}"
MASTER_PORT="${MASTER_PORT:-29517}"
LOG_RANK="${LOG_RANK:-0}"
PYTHON="${PYTHON:-python}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/ckpts/action_aware_transition_bs8_f4_${RUN_TAG}}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if ! "${PYTHON}" -c "import lerobot" >/dev/null 2>&1; then
    echo "ERROR: lerobot is not installed in $("${PYTHON}" -c 'import sys; print(sys.executable)')" >&2
    exit 1
fi

args=(
    --config-name robotwin_contrastive_align
    --save-root "${SAVE_ROOT}"
    --lambda-action-chunk "${LAMBDA_ACTION_CHUNK:-0.1}"
    --lambda-soft-contrastive "${LAMBDA_SOFT_CONTRASTIVE:-0.05}"
    --aux-warmup-steps "${AUX_WARMUP_STEPS:-250}"
    --aux-max-primary-fraction "${AUX_MAX_PRIMARY_FRACTION:-0.15}"
    --soft-positive-mix "${SOFT_POSITIVE_MIX:-0.5}"
)
if [[ -n "${RESUME_FROM:-}" ]]; then args+=(--resume-from "${RESUME_FROM}"); fi
if [[ -n "${NUM_STEPS:-}" ]]; then args+=(--num-steps "${NUM_STEPS}"); fi
if [[ -n "${SAVE_INTERVAL:-}" ]]; then args+=(--save-interval "${SAVE_INTERVAL}"); fi
if [[ -n "${BATCH_SIZE:-}" ]]; then args+=(--batch-size "${BATCH_SIZE}"); fi
if [[ "${ENABLE_WANDB}" == "1" ]]; then
    : "${WANDB_API_KEY:?Set WANDB_API_KEY when ENABLE_WANDB=1}"
    export WANDB_PROJECT="${WANDB_PROJECT:-va_robotwin_action_aware}"
    export WANDB_RUN_NAME="${WANDB_RUN_NAME:-action_aware_bs8_f4_${RUN_TAG}}"
    export WANDB_MODE="${WANDB_MODE:-online}"
    args+=(--enable-wandb)
else
    args+=(--disable-wandb)
fi
args+=("$@")

mkdir -p "${SAVE_ROOT}"
echo "Action-aware training: GPUs=${NGPU}, per_gpu_batch=${BATCH_SIZE:-8}, latent_frames=4"
echo "Output: ${SAVE_ROOT}"

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NGPU}" \
    --local-ranks-filter="${LOG_RANK}" \
    --master_port="${MASTER_PORT}" \
    --tee=3 \
    -m wan_va.train_action_aware_align \
    "${args[@]}"
