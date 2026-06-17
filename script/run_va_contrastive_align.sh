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
if [ $# -ne 0 ]; then
    args+=("$@")
fi

export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
python -m torch.distributed.run \
    --nproc_per_node="${NGPU}" \
    --local-ranks-filter="${LOG_RANK}" \
    --master_port "${MASTER_PORT}" \
    --tee 3 \
    -m wan_va.train_contrastive_align "${args[@]}"
