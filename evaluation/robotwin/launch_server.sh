#!/usr/bin/env bash
set -euo pipefail

START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
CONFIG_NAME=${CONFIG_NAME:-robotwin}
PRETRAINED_MODEL=${1:-${PRETRAINED_MODEL:-${MODEL_PATH:-}}}

save_root=${SERVER_SAVE_ROOT:-visualization/}
mkdir -p $save_root

server_args=(
    --config-name "$CONFIG_NAME"
    --port "$START_PORT"
    --save_root "$save_root"
)

if [ -n "$PRETRAINED_MODEL" ]; then
    server_args+=(--pretrained-model "$PRETRAINED_MODEL")
fi

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    "${server_args[@]}"
