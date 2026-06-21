#!/usr/bin/env bash

set -euo pipefail

umask 007

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29501"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"} # robotwin_train, libero_train, demo_train
PYTHON=${PYTHON:-python}

ENABLE_WANDB=${ENABLE_WANDB:-"0"}
WANDB_BASE_URL=${WANDB_BASE_URL:-"https://api.wandb.ai"}
WANDB_ENTITY=${WANDB_ENTITY:-${WANDB_TEAM_NAME:-}}
WANDB_PROJECT=${WANDB_PROJECT:-"va_robotwin_train"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-""}

overrides=()
if [ -n "${SAVE_ROOT:-}" ]; then
    overrides+=(--save-root "${SAVE_ROOT}")
fi
if [ -n "${RESUME_FROM:-}" ]; then
    overrides+=(--resume-from "${RESUME_FROM}")
fi
if [ -n "${DATASET_PATH:-}" ]; then
    overrides+=(--dataset-path "${DATASET_PATH}")
fi
if [ -n "${EMPTY_EMB_PATH:-}" ]; then
    overrides+=(--empty-emb-path "${EMPTY_EMB_PATH}")
fi
if [ -n "${PRETRAINED_MODEL:-}" ]; then
    overrides+=(--pretrained-model "${PRETRAINED_MODEL}")
fi
if [ -n "${NUM_STEPS:-}" ]; then
    overrides+=(--num-steps "${NUM_STEPS}")
fi
if [ $# -ne 0 ]; then
    overrides+=("$@")
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

if [ "${ENABLE_WANDB}" = "1" ]; then
    : "${WANDB_API_KEY:?Set WANDB_API_KEY when ENABLE_WANDB=1}"
    export WANDB_API_KEY
    export WANDB_BASE_URL
    export WANDB_PROJECT
    export WANDB_MODE=${WANDB_MODE:-"online"}
    if [ -n "${WANDB_ENTITY}" ]; then
        export WANDB_ENTITY
    fi
    if [ -n "${WANDB_RUN_NAME}" ]; then
        export WANDB_RUN_NAME
    fi
    overrides+=(--enable-wandb)
else
    overrides+=(--disable-wandb)
fi

set -x

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
    -m wan_va.train \
    --config-name "${CONFIG_NAME}" \
    "${overrides[@]}"
