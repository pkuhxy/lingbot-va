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

for egl_vendor_dir in /etc/glvnd/egl_vendor.d /usr/share/glvnd/egl_vendor.d; do
if [ ! -d "${egl_vendor_dir}" ]; then
    if [ "$(id -u)" -eq 0 ]; then
        mkdir -p "${egl_vendor_dir}"
        echo "Created missing EGL vendor directory: ${egl_vendor_dir}"
    else
        echo "Missing EGL vendor directory: ${egl_vendor_dir}" >&2
        echo "SAPIEN needs the GLVND EGL vendor directory to initialize rendering." >&2
        echo "Create it or install Vulkan/GLVND packages on the server:" >&2
        echo "  sudo mkdir -p /etc/glvnd/egl_vendor.d /usr/share/glvnd/egl_vendor.d" >&2
        echo "  sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools libegl1 libglvnd0" >&2
        exit 1
    fi
fi
done

FFMPEG_BINARY=${FFMPEG_BINARY:-ffmpeg}
if ! command -v "${FFMPEG_BINARY}" >/dev/null 2>&1; then
    echo "Missing ffmpeg executable: ${FFMPEG_BINARY}" >&2
    exit 1
fi
export FFMPEG_BINARY

save_root=${1:-"${PROJECT_ROOT}/results/rt_c2r"}
seed=${2:-0}
test_num=${3:-100}

policy_name=${POLICY_NAME:-ACT}
train_config_name=${TRAIN_CONFIG_NAME:-0}
model_name=${MODEL_NAME:-0}
start_port=${START_PORT:-29056}
num_gpus=${NUM_GPUS:-1}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
SAVE_VISUALIZATION=${SAVE_VISUALIZATION:-False}
VALIDATED_MANIFEST_DIR=${VALIDATED_MANIFEST_DIR:-"${PROJECT_ROOT}/evaluation/robotwin/rt_c2r_validated_manifests"}
if [ -f "${VALIDATED_MANIFEST_DIR}/rt_c2r_easy.jsonl" ]; then
    DEFAULT_MANIFEST_DIR="${VALIDATED_MANIFEST_DIR}"
else
    DEFAULT_MANIFEST_DIR="${PROJECT_ROOT}/evaluation/robotwin/rt_c2r_manifests"
fi
MANIFEST_DIR=${MANIFEST_DIR:-"${DEFAULT_MANIFEST_DIR}"}
if [ -z "${STRICT_SEED_MANIFEST+x}" ]; then
    if [ "${MANIFEST_DIR}" = "${VALIDATED_MANIFEST_DIR}" ]; then
        STRICT_SEED_MANIFEST=True
    else
        STRICT_SEED_MANIFEST=False
    fi
fi
SPLITS=${SPLITS:-"easy background light clutter height hard"}
REGENERATE_MANIFESTS=${REGENERATE_MANIFESTS:-False}

if [ "${REGENERATE_MANIFESTS}" = "True" ] || [ ! -f "${MANIFEST_DIR}/rt_c2r_easy.jsonl" ]; then
    python "${SCRIPT_DIR}/generate_rt_c2r_benchmark.py" \
        --manifest-dir "${MANIFEST_DIR}" \
        --episodes-per-task "${test_num}"
fi

python "${SCRIPT_DIR}/generate_rt_c2r_benchmark.py" \
    --skip-manifests \
    --write-task-configs \
    --robotwin-root "${ROBOTWIN_ROOT}"

task_names=(
  stack_bowls_three
  handover_block
  hanging_mug
  scan_object
  lift_pot
  put_object_cabinet
  stack_blocks_three
  place_shoe
  adjust_bottle
  place_mouse_pad
  dump_bin_bigbin
  move_pillbottle_pad
  pick_dual_bottles
  shake_bottle
  place_fan
  turn_switch
  shake_bottle_horizontally
  place_container_plate
  rotate_qrcode
  place_object_stand
  put_bottles_dustbin
  move_stapler_pad
  place_burger_fries
  place_bread_basket
  pick_diverse_bottles
  open_microwave
  beat_block_hammer
  press_stapler
  click_bell
  move_playingcard_away
  open_laptop
  move_can_pot
  stack_bowls_two
  place_a2b_right
  stamp_seal
  place_object_basket
  handover_mic
  place_bread_skillet
  stack_blocks_two
  place_cans_plasticbox
  click_alarmclock
  blocks_ranking_size
  place_phone_stand
  place_can_basket
  place_object_scale
  place_a2b_left
  grab_roller
  place_dual_shoes
  place_empty_cup
  blocks_ranking_rgb
)

log_dir="${PROJECT_ROOT}/logs/rt_c2r"
mkdir -p "$log_dir"

batch_time=$(date +%Y%m%d_%H%M%S)
pid_file="${PROJECT_ROOT}/pids_rt_c2r_${batch_time}.txt"
> "$pid_file"

echo "save_root=${save_root}"
echo "seed=${seed}"
echo "test_num=${test_num}"
echo "num_tasks=${#task_names[@]}"
echo "splits=${SPLITS}"
echo "num_gpus=${num_gpus}"
echo "server_host=${SERVER_HOST}"
echo "start_port=${start_port}"

run_one_task() {
    local split=$1
    local task_id=$2
    local task_name=$3
    local gpu_id=$4
    local port=$5
    local log_file=$6

    local task_config="rt_c2r_${split}"
    local split_manifest="${MANIFEST_DIR}/${task_config}.jsonl"
    local split_save_root="${save_root}/${split}"

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    python -m evaluation.robotwin.eval_polict_client_openpi --config policy/$policy_name/deploy_policy.yml \
        --overrides \
        --task_name ${task_name} \
        --task_config ${task_config} \
        --train_config_name ${train_config_name} \
        --model_name ${model_name} \
        --ckpt_setting ${model_name} \
        --seed ${seed} \
        --policy_name ${policy_name} \
        --save_root ${split_save_root} \
        --seed_manifest ${split_manifest} \
        --strict_seed_manifest ${STRICT_SEED_MANIFEST} \
        --split_name ${split} \
        --video_guidance_scale 5 \
        --action_guidance_scale 1 \
        --save_visualization ${SAVE_VISUALIZATION} \
        --test_num ${test_num} \
        --host ${SERVER_HOST} \
        --port ${port} > "$log_file" 2>&1
}

total=${#task_names[@]}
for split in ${SPLITS}; do
    echo -e "\033[32mStarting RT-C2R split: ${split}\033[0m"
    for (( batch_start=0; batch_start<total; batch_start+=num_gpus )); do
        batch_end=$(( batch_start + num_gpus ))
        if (( batch_end > total )); then
            batch_end=${total}
        fi

        echo -e "\033[32mLaunching ${split} batch $(( batch_start / num_gpus + 1 )): tasks ${batch_start}..$(( batch_end - 1 ))\033[0m"
        pids=()

        for (( i=batch_start; i<batch_end; i++ )); do
            task_name="${task_names[$i]}"
            gpu_id=$(( (i - batch_start) % num_gpus ))
            port=$(( start_port + gpu_id ))
            log_file="${log_dir}/${split}_${i}_${task_name}_${batch_time}.log"

            echo -e "\033[33m[${split} Task $i/$(( total - 1 ))] ${task_name}: GPU ${gpu_id}, PORT ${port}, Log ${log_file}\033[0m"
            run_one_task "$split" "$i" "$task_name" "$gpu_id" "$port" "$log_file" &
            pid=$!
            pids+=("$pid")
            echo "${pid}" >> "$pid_file"
        done

        failed=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=1
            fi
        done

        if (( failed )); then
            echo "At least one task failed in split ${split}. See logs under ${log_dir}." >&2
            exit 1
        fi
    done
done

python "${SCRIPT_DIR}/calc_rt_c2r_stat.py" "${save_root}" --output "${save_root}/summary.json"

echo -e "\033[32mRT-C2R benchmark completed.\033[0m"
echo "PID history saved to ${pid_file}"
