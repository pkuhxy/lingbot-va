#!/usr/bin/env bash

set -euo pipefail
set -x

umask 007

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29503"}
LOG_RANK=${LOG_RANK:-"0"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_contrastive_align"}
SAVE_ROOT=${SAVE_ROOT:-""}
RESUME_FROM=${RESUME_FROM:-""}
LAMBDA_ALIGN=${LAMBDA_ALIGN:-""}
LAMBDA_ACTION_RECON=${LAMBDA_ACTION_RECON:-""}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
PYTHON=${PYTHON:-python}
ENABLE_WANDB=${ENABLE_WANDB:-"0"}
WANDB_BASE_URL=${WANDB_BASE_URL:-"https://api.wandb.ai"}
WANDB_PROJECT=${WANDB_PROJECT:-"va_robotwin_contrastive_align"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-""}

args=(--config-name "${CONFIG_NAME}")
if [ -n "${SAVE_ROOT}" ]; then
    args+=(--save-root "${SAVE_ROOT}")
fi
if [ -n "${RESUME_FROM}" ]; then
    args+=(--resume-from "${RESUME_FROM}")
fi
if [ -n "${LAMBDA_ALIGN}" ]; then
    args+=(--lambda-align "${LAMBDA_ALIGN}")
fi
if [ -n "${LAMBDA_ACTION_RECON}" ]; then
    args+=(--lambda-action-recon "${LAMBDA_ACTION_RECON}")
fi
if [ "${ENABLE_WANDB}" = "1" ]; then
    : "${WANDB_API_KEY:?Set WANDB_API_KEY when ENABLE_WANDB=1}"
    : "${WANDB_TEAM_NAME:?Set WANDB_TEAM_NAME when ENABLE_WANDB=1}"
    export WANDB_API_KEY
    export WANDB_BASE_URL
    export WANDB_TEAM_NAME
    export WANDB_PROJECT
    if [ -n "${WANDB_RUN_NAME}" ]; then
        export WANDB_RUN_NAME
    fi
    "${PYTHON}" - <<'PY'
import os
import wandb

wandb.login(
    host=os.environ["WANDB_BASE_URL"],
    key=os.environ["WANDB_API_KEY"],
    relogin=True,
)
print("WandB initialized for project:", os.environ["WANDB_PROJECT"])
PY
    args+=(--enable-wandb)
else
    args+=(--disable-wandb)
fi
if [ $# -ne 0 ]; then
    args+=("$@")
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
if ! "${PYTHON}" -c "import lerobot" >/dev/null 2>&1; then
    echo "ERROR: lerobot is not installed in this Python environment: $(${PYTHON} -c 'import sys; print(sys.executable)')" >&2
    echo "Run: PYTHON=${PYTHON} bash script/install_posttrain_deps.sh" >&2
    exit 1
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NGPU}" \
    --local-ranks-filter="${LOG_RANK}" \
    --master_port "${MASTER_PORT}" \
    --tee 3 \
    -m wan_va.train_contrastive_align "${args[@]}"
