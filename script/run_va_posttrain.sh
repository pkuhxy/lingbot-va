#!/usr/bin/env bash

set -euo pipefail
set -x

umask 007
 
NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_contrastive_align"} # robotwin_contrastive_align, robotwin_train, libero_train
PYTHON=${PYTHON:-python}
ENABLE_WANDB=${ENABLE_WANDB:-"0"}
WANDB_BASE_URL=${WANDB_BASE_URL:-"https://api.wandb.ai"}
WANDB_PROJECT=${WANDB_PROJECT:-"va_robotwin_contrastive_align"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-""}
WANDB_ENTITY=${WANDB_ENTITY:-${WANDB_TEAM_NAME:-""}}
WANDB_MODE=${WANDB_MODE:-"online"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
if [ "${ENABLE_WANDB}" = "1" ]; then
    : "${WANDB_API_KEY:?Set WANDB_API_KEY when ENABLE_WANDB=1}"
    export WANDB_API_KEY
    export WANDB_BASE_URL
    export WANDB_PROJECT
    export WANDB_MODE
    if [ -n "${WANDB_RUN_NAME}" ]; then
        export WANDB_RUN_NAME
    fi
    if [ -n "${WANDB_ENTITY}" ]; then
        export WANDB_ENTITY
        export WANDB_TEAM_NAME="${WANDB_ENTITY}"
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
print("WandB entity:", os.environ.get("WANDB_ENTITY", "<default>"))
PY
    overrides="${overrides} --enable-wandb"
else
    overrides="${overrides} --disable-wandb"
fi

if ! "${PYTHON}" -c "import lerobot" >/dev/null 2>&1; then
    echo "ERROR: lerobot is not installed in this Python environment: $(${PYTHON} -c 'import sys; print(sys.executable)')" >&2
    echo "Run: PYTHON=${PYTHON} bash script/install_posttrain_deps.sh" >&2
    exit 1
fi

PYTORCH_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train_contrastive_align --config-name ${config_name} $overrides
