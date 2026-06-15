#!/usr/bin/env bash

set -euo pipefail
set -x

umask 007

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29502"}
LOG_RANK=${LOG_RANK:-"0"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_latent_posttrain"}
DATASET_PATH=${DATASET_PATH:-""}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"/mnt/data/share/checkpoints/robbyant/lingbot-va-base"}
SAVE_ROOT=${SAVE_ROOT:-"./train_out_wam_latent"}
HIDDEN_SOURCE=${HIDDEN_SOURCE:-""}
ATTN_MODE=${ATTN_MODE:-""}
NUM_STEPS=${NUM_STEPS:-""}

if [ -z "${DATASET_PATH}" ]; then
    echo "ERROR: DATASET_PATH is required."
    echo "Example:"
    echo "  DATASET_PATH=/path/to/robotwin-clean-and-aug-lerobot NGPU=8 bash script/run_wam_latent_posttrain_stage1.sh"
    exit 1
fi

if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: DATASET_PATH does not exist or is not a directory: ${DATASET_PATH}"
    exit 1
fi
DATASET_PATH="$(cd "${DATASET_PATH}" && pwd)"

if [ ! -f "${DATASET_PATH}/empty_emb.pt" ]; then
    echo "ERROR: ${DATASET_PATH}/empty_emb.pt is missing."
    echo "DATASET_PATH should usually point to the dataset top-level directory, not only a clean/aug subdirectory."
    exit 1
fi

if [ -n "${PRETRAINED_MODEL}" ] && [ ! -d "${PRETRAINED_MODEL}/transformer" ]; then
    echo "ERROR: pretrained transformer directory is missing: ${PRETRAINED_MODEL}/transformer"
    exit 1
fi

if ! find "${DATASET_PATH}" -path '*/meta/info.json' -print -quit | grep -q .; then
    echo "ERROR: no LeRobot meta/info.json found under ${DATASET_PATH}"
    exit 1
fi

if ! find "${DATASET_PATH}" -type d -name latents -print -quit | grep -q .; then
    echo "ERROR: no latents/ directory found under ${DATASET_PATH}"
    exit 1
fi

overrides=""
overrides="${overrides} --dataset-path ${DATASET_PATH}"
if [ -n "${PRETRAINED_MODEL}" ]; then
    overrides="${overrides} --pretrained-model ${PRETRAINED_MODEL}"
fi
overrides="${overrides} --save-root ${SAVE_ROOT}"
if [ -n "${HIDDEN_SOURCE}" ]; then
    overrides="${overrides} --hidden-source ${HIDDEN_SOURCE}"
fi
if [ -n "${ATTN_MODE}" ]; then
    overrides="${overrides} --attn-mode ${ATTN_MODE}"
fi
if [ -n "${NUM_STEPS}" ]; then
    overrides="${overrides} --num-steps ${NUM_STEPS}"
fi
if [ $# -ne 0 ]; then
    overrides="${overrides} $*"
fi

export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m torch.distributed.run \
    --nproc_per_node=${NGPU} \
    --local-ranks-filter=${LOG_RANK} \
    --master_port ${MASTER_PORT} \
    --tee 3 \
    -m wan_va.train_latent_posttrain --config-name ${CONFIG_NAME} ${overrides}
