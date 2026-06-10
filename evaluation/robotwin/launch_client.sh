#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}
export ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-"${PROJECT_ROOT}/RoboTwin"}
export PYTHONPATH="${PROJECT_ROOT}:${ROBOTWIN_ROOT}:${PYTHONPATH:-}"

if [ ! -d "${ROBOTWIN_ROOT}/envs" ]; then
    echo "RoboTwin root not found: ${ROBOTWIN_ROOT}" >&2
    echo "Set ROBOTWIN_ROOT=/path/to/RoboTwin before running this script." >&2
    exit 1
fi

if [ ! -d /etc/glvnd/egl_vendor.d ]; then
    echo "Missing EGL vendor directory: /etc/glvnd/egl_vendor.d" >&2
    echo "SAPIEN needs the GLVND EGL vendor directory to initialize rendering." >&2
    echo "Install Vulkan/GLVND packages or create the directory on the server:" >&2
    echo "  sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools libegl1 libglvnd0" >&2
    echo "  sudo mkdir -p /etc/glvnd/egl_vendor.d" >&2
    exit 1
fi

FFMPEG_BINARY=${FFMPEG_BINARY:-ffmpeg}
if ! command -v "${FFMPEG_BINARY}" >/dev/null 2>&1; then
    echo "Missing ffmpeg executable: ${FFMPEG_BINARY}" >&2
    echo "RoboTwin writes evaluation videos by launching an ffmpeg command." >&2
    echo "Install it on the server:" >&2
    echo "  conda install -c conda-forge ffmpeg -y" >&2
    echo "or set FFMPEG_BINARY=/path/to/ffmpeg before running this script." >&2
    exit 1
fi
export FFMPEG_BINARY

task_groups=(
  "stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe"
  "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch"
  "shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket"
  "pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot"
  "stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox"
  "click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes"
  "place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb"
)

save_root=${1:-'./results'}
task_name=${2:-"adjust_bottle"}

policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=0
PORT=${PORT:-29056}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
SAVE_VISUALIZATION=${SAVE_VISUALIZATION:-True}

PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m evaluation.robotwin.eval_polict_client_openpi --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --save_root ${save_root} \
    --video_guidance_scale 5 \
    --action_guidance_scale 1 \
    --save_visualization ${SAVE_VISUALIZATION} \
    --test_num 100 \
    --host ${SERVER_HOST} \
    --port ${PORT}
