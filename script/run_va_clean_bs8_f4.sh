#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

NGPU="${NGPU:-2}"
MASTER_PORT="${MASTER_PORT:-29507}"
LOG_RANK="${LOG_RANK:-0}"
CONFIG_NAME="robotwin_train_bs8_f4"
PYTHON="${PYTHON:-python}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if ! "${PYTHON}" -c "import lerobot" >/dev/null 2>&1; then
    echo "ERROR: lerobot is not installed in: $("${PYTHON}" -c 'import sys; print(sys.executable)')" >&2
    exit 1
fi

args=(--config-name "${CONFIG_NAME}")
if [[ -n "${SAVE_ROOT:-}" ]]; then args+=(--save-root "${SAVE_ROOT}"); fi
if [[ -n "${RESUME_FROM:-}" ]]; then args+=(--resume-from "${RESUME_FROM}"); fi
if [[ "${ENABLE_WANDB}" == "1" ]]; then
    : "${WANDB_API_KEY:?Set WANDB_API_KEY when ENABLE_WANDB=1}"
    export WANDB_PROJECT="${WANDB_PROJECT:-va_robotwin_train_bs8_f4}"
    export WANDB_RUN_NAME="${WANDB_RUN_NAME:-clean_bs8_f4}"
    export WANDB_MODE="${WANDB_MODE:-online}"
    args+=(--enable-wandb)
else
    args+=(--disable-wandb)
fi
args+=("$@")

echo "Clean ablation training: per_gpu_batch=8 latent_frames=4 GPUs=${NGPU} effective_global_batch=$((8 * NGPU))"
echo "Output: ${SAVE_ROOT:-/mnt/data/users/xianyi/EmbodyAi/lingbot-va/ckpts/train_out_robotwin_clean_bs8_f4}"

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NGPU}" \
    --local-ranks-filter="${LOG_RANK}" \
    --master_port="${MASTER_PORT}" \
    --tee=3 \
    -m wan_va.train \
    "${args[@]}"
