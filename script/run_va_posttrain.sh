#!/usr/bin/bash

set -x

umask 007
 
NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_contrastive_align"} # robotwin_contrastive_align, robotwin_train, libero_train
PYTHON=${PYTHON:-python}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

export WANDB_API_KEY="your key"
export WANDB_BASE_URL="your url"
export WANDB_TEAM_NAME="your team name"
export WANDB_PROJECT="your project"

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
