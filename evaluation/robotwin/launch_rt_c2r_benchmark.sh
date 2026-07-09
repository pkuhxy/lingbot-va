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
policy_config=${POLICY_CONFIG:-"${ROBOTWIN_ROOT}/policy/${policy_name}/deploy_policy.yml"}
start_port=${START_PORT:-29056}
master_port=${MASTER_PORT:-29061}
num_gpus=${NUM_GPUS:-1}
GPU_IDS=${GPU_IDS:-}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
SAVE_VISUALIZATION=${SAVE_VISUALIZATION:-False}
SPLITS=${SPLITS:-"easy background light clutter height hard"}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-${MODEL_PATH:-}}
AUTO_START_SERVER=${AUTO_START_SERVER:-False}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}
SERVER_LOG_DIR=${SERVER_LOG_DIR:-"${save_root}/server_logs"}
SERVER_SAVE_ROOT=${SERVER_SAVE_ROOT:-"${save_root}/server_visualization"}
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
REGENERATE_MANIFESTS=${REGENERATE_MANIFESTS:-False}
ALLOW_MANIFEST_EXPERT_FALLBACK=${ALLOW_MANIFEST_EXPERT_FALLBACK:-True}

if [ ! -f "${policy_config}" ]; then
    echo "Policy config not found: ${policy_config}" >&2
    echo "Set POLICY_CONFIG=/path/to/deploy_policy.yml or POLICY_NAME to a policy under ${ROBOTWIN_ROOT}/policy." >&2
    exit 1
fi

if [ "${STRICT_SEED_MANIFEST}" = "True" ]; then
    for split in ${SPLITS}; do
        split_manifest="${MANIFEST_DIR}/rt_c2r_${split}.jsonl"
        if [ ! -f "${split_manifest}" ]; then
            echo "Missing strict RT-C2R seed manifest: ${split_manifest}" >&2
            echo "Run generate_rt_c2r_validated_manifests.py first, or set MANIFEST_DIR to the validated manifest directory." >&2
            exit 1
        fi
        if grep -Evq '^[[:space:]]*$|"expert_validated"[[:space:]]*:[[:space:]]*true' "${split_manifest}"; then
            echo "Strict RT-C2R manifest does not look expert-validated: ${split_manifest}" >&2
            echo "Refusing to run with raw candidate manifests. Set MANIFEST_DIR to rt_c2r_validated_manifests." >&2
            exit 1
        fi
    done
elif [ "${REGENERATE_MANIFESTS}" = "True" ] || [ ! -f "${MANIFEST_DIR}/rt_c2r_easy.jsonl" ]; then
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

log_dir=${LOG_DIR:-"${PROJECT_ROOT}/logs/rt_c2r"}
mkdir -p "$log_dir"
pid_dir=${PID_DIR:-"${PROJECT_ROOT}"}
mkdir -p "$pid_dir"

if [ -n "$GPU_IDS" ]; then
    IFS=',' read -r -a gpu_ids <<< "$GPU_IDS"
    num_gpus=${#gpu_ids[@]}
else
    gpu_ids=()
    for (( i=0; i<num_gpus; i++ )); do
        gpu_ids+=("$i")
    done
fi

batch_time=$(date +%Y%m%d_%H%M%S)
pid_file="${pid_dir}/pids_rt_c2r_${batch_time}.txt"
> "$pid_file"

echo "save_root=${save_root}"
echo "seed=${seed}"
echo "test_num=${test_num}"
echo "num_tasks=${#task_names[@]}"
echo "splits=${SPLITS}"
echo "num_gpus=${num_gpus}"
echo "gpu_ids=${gpu_ids[*]}"
echo "server_host=${SERVER_HOST}"
echo "start_port=${start_port}"
echo "master_port=${master_port}"
echo "manifest_dir=${MANIFEST_DIR}"
echo "strict_seed_manifest=${STRICT_SEED_MANIFEST}"
echo "allow_manifest_expert_fallback=${ALLOW_MANIFEST_EXPERT_FALLBACK}"
echo "policy_config=${policy_config}"
echo "log_dir=${log_dir}"
echo "pid_file=${pid_file}"
echo "auto_start_server=${AUTO_START_SERVER}"

port_is_open() {
    local port=$1
    (echo > "/dev/tcp/${SERVER_HOST}/${port}") >/dev/null 2>&1
}

wait_for_server() {
    local port=$1
    local pid=$2
    local log_file=$3
    local start_ts
    local now_ts
    start_ts=$(date +%s)

    while true; do
        if port_is_open "${port}"; then
            return 0
        fi

        if ! kill -0 "${pid}" >/dev/null 2>&1; then
            echo "Server exited before port ${port} became ready. Log: ${log_file}" >&2
            tail -n 120 "${log_file}" >&2 || true
            exit 1
        fi

        now_ts=$(date +%s)
        if (( now_ts - start_ts >= SERVER_READY_TIMEOUT )); then
            echo "Timed out waiting for server port ${port}. Log: ${log_file}" >&2
            tail -n 120 "${log_file}" >&2 || true
            exit 1
        fi

        sleep 5
    done
}

server_pids=()
cleanup_servers() {
    local status=$?
    trap - EXIT INT TERM
    if ((${#server_pids[@]} > 0)); then
        echo "Stopping RT-C2R server processes: ${server_pids[*]}"
        for pid in "${server_pids[@]}"; do
            if kill -0 "${pid}" >/dev/null 2>&1; then
                kill "${pid}" >/dev/null 2>&1 || true
            fi
        done
        for pid in "${server_pids[@]}"; do
            wait "${pid}" >/dev/null 2>&1 || true
        done
    fi
    exit "${status}"
}

if [ "${AUTO_START_SERVER}" = "True" ]; then
    if [ -z "${PRETRAINED_MODEL}" ]; then
        echo "AUTO_START_SERVER=True requires PRETRAINED_MODEL=/path/to/checkpoint." >&2
        exit 1
    fi
    if [ ! -e "${PRETRAINED_MODEL}" ]; then
        echo "Pretrained model path not found: ${PRETRAINED_MODEL}" >&2
        exit 1
    fi

    for (( i=0; i<num_gpus; i++ )); do
        port=$(( start_port + i ))
        if port_is_open "${port}"; then
            echo "Server port already in use: ${SERVER_HOST}:${port}" >&2
            echo "Stop the existing server or choose another START_PORT." >&2
            exit 1
        fi
    done

    mkdir -p "${SERVER_LOG_DIR}" "${SERVER_SAVE_ROOT}"
    for (( i=0; i<num_gpus; i++ )); do
        gpu_id="${gpu_ids[$i]}"
        port=$(( start_port + i ))
        current_master_port=$(( master_port + i ))
        server_log="${SERVER_LOG_DIR}/server_${i}_${batch_time}.log"
        echo "Starting RT-C2R server ${i}: ckpt=${PRETRAINED_MODEL}, GPU=${gpu_id}, PORT=${port}, Log=${server_log}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        START_PORT="${port}" \
        MASTER_PORT="${current_master_port}" \
        SERVER_SAVE_ROOT="${SERVER_SAVE_ROOT}" \
        bash "${SCRIPT_DIR}/launch_server.sh" "${PRETRAINED_MODEL}" > "${server_log}" 2>&1 &
        server_pids+=("$!")
    done

    trap cleanup_servers EXIT INT TERM
    for (( i=0; i<num_gpus; i++ )); do
        port=$(( start_port + i ))
        wait_for_server "${port}" "${server_pids[$i]}" "${SERVER_LOG_DIR}/server_${i}_${batch_time}.log"
    done
fi

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
    python -m evaluation.robotwin.eval_polict_client_openpi --config "${policy_config}" \
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
        --allow_manifest_expert_fallback ${ALLOW_MANIFEST_EXPERT_FALLBACK} \
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
        batch_logs=()

        for (( i=batch_start; i<batch_end; i++ )); do
            task_name="${task_names[$i]}"
            gpu_slot=$(( (i - batch_start) % num_gpus ))
            gpu_id="${gpu_ids[$gpu_slot]}"
            port=$(( start_port + gpu_slot ))
            log_file="${log_dir}/${split}_${i}_${task_name}_${batch_time}.log"

            echo -e "\033[33m[${split} Task $i/$(( total - 1 ))] ${task_name}: GPU ${gpu_id}, PORT ${port}, Log ${log_file}\033[0m"
            run_one_task "$split" "$i" "$task_name" "$gpu_id" "$port" "$log_file" &
            pid=$!
            pids+=("$pid")
            batch_logs+=("$log_file")
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
            echo "Failure summary from this batch:" >&2
            for batch_log in "${batch_logs[@]}"; do
                echo "===== ${batch_log}" >&2
                if grep -Eq 'Traceback|RuntimeError|Exception|Error|KeyboardInterrupt|ConnectionRefused' "${batch_log}"; then
                    grep -En 'Traceback|RuntimeError|Exception|Error|KeyboardInterrupt|ConnectionRefused' "${batch_log}" | tail -n 40 >&2 || true
                else
                    tail -n 40 "${batch_log}" >&2 || true
                fi
            done
            exit 1
        fi
    done
done

python "${SCRIPT_DIR}/calc_rt_c2r_stat.py" "${save_root}" \
    --output "${save_root}/summary.json" \
    --csv-output "${save_root}/summary.csv"

echo -e "\033[32mRT-C2R benchmark completed.\033[0m"
echo "PID history saved to ${pid_file}"
