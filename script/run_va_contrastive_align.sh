#!/usr/bin/env bash

set -euo pipefail

umask 007

# Fill this in on the training machine. Avoid committing a real API key.
export WANDB_API_KEY="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"
export WANDB_BASE_URL="https://api.wandb.ai"
# Leave empty to use the default W&B entity for this API key.
# Fill with the exact W&B username/team slug only when you need a specific entity.
export WANDB_ENTITY=""
unset WANDB_TEAM_NAME
export WANDB_PROJECT="va_robotwin_contrastive_align"
export WANDB_RUN_NAME="exp1_clean"
export WANDB_MODE="online"

set -x

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29503"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_contrastive_align"}
PYTHON=${PYTHON:-python}

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
    -m wan_va.train_contrastive_align \
    --config-name "${CONFIG_NAME}" \
    --enable-wandb \
    "$@"
