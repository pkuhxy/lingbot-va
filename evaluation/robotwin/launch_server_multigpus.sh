#!/usr/bin/env bash
set -euo pipefail

START_PORT=${START_PORT:-29556}
MASTER_PORT=${MASTER_PORT:-29661}
NUM_GPUS=${NUM_GPUS:-8}
GPU_IDS=${GPU_IDS:-}
CONFIG_NAME=${CONFIG_NAME:-robotwin}
PRETRAINED_MODEL=${1:-${PRETRAINED_MODEL:-${MODEL_PATH:-}}}
LOG_DIR='./logs'
mkdir -p $LOG_DIR

save_root=${SERVER_SAVE_ROOT:-./visualization/}
mkdir -p $save_root

batch_time=$(date +%Y%m%d_%H%M%S)

if [ -n "$GPU_IDS" ]; then
    IFS=',' read -r -a gpu_ids <<< "$GPU_IDS"
    NUM_GPUS=${#gpu_ids[@]}
else
    gpu_ids=()
    for (( i=0; i<NUM_GPUS; i++ )); do
        gpu_ids+=("$i")
    done
fi

for (( i=0; i<NUM_GPUS; i++ )); do
    GPU_ID="${gpu_ids[$i]}"
    CURRENT_PORT=$((START_PORT + i))
    CURRENT_MASTER_PORT=$((MASTER_PORT + i))

    LOG_FILE="${LOG_DIR}/server_${i}_${batch_time}.log"
    echo "[Server ${i}] GPU: ${GPU_ID} | PORT: ${CURRENT_PORT} | MASTER_PORT: ${CURRENT_MASTER_PORT} | Log: ${LOG_FILE}"

    server_args=(
        --config-name "$CONFIG_NAME"
        --save_root "$save_root"
        --port "$CURRENT_PORT"
    )
    if [ -n "$PRETRAINED_MODEL" ]; then
        server_args+=(--pretrained-model "$PRETRAINED_MODEL")
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID  \
    nohup python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port $CURRENT_MASTER_PORT \
        wan_va/wan_va_server.py \
        "${server_args[@]}" > "$LOG_FILE" 2>&1 &
    sleep 2;
done

echo "All ${NUM_GPUS} instances have been launched in the background."
wait
