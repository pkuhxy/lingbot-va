#!/usr/bin/env bash
set -euo pipefail

# Bulk-download WAM/VAM checkpoints from Hugging Face.
#
# Usage:
#   bash script/download_wam_weights.sh
#   SAVE_ROOT=/mnt/data/share/checkpoints bash script/download_wam_weights.sh
#   INCLUDE_BACKBONES=0 bash script/download_wam_weights.sh
#   INCLUDE_VLA_BASELINES=1 bash script/download_wam_weights.sh
#   DRY_RUN=1 bash script/download_wam_weights.sh
#
# Notes:
#   - Some repos are very large or gated. Run `hf auth login` first if needed.
#   - The default endpoint uses hf-mirror.com, matching the example script.
#   - Entries are formatted as: repo_type|repo_id|revision

SAVE_ROOT="${SAVE_ROOT:-/mnt/data/share/checkpoints}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
INCLUDE_BACKBONES="${INCLUDE_BACKBONES:-1}"
INCLUDE_VLA_BASELINES="${INCLUDE_VLA_BASELINES:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"

export HF_ENDPOINT
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

if command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli download)
else
  if [[ "$DRY_RUN" == "1" ]]; then
    HF_CLI=(hf download)
  else
  echo "Cannot find hf or huggingface-cli. Install with:"
  echo "  pip install 'huggingface_hub[cli]' hf_transfer"
  exit 1
  fi
fi

CORE_WAM_REPOS=(
  # LingBot-VA
  "model|robbyant/lingbot-va-base|"
  "model|robbyant/lingbot-va-posttrain-robotwin|"
  "model|robbyant/lingbot-va-posttrain-libero-long|"

  # Motus
  "model|motus-robotics/Motus_Wan2_2_5B_pretrain|"
  "model|motus-robotics/Motus|"
  "model|motus-robotics/Motus_robotwin2|"

  # DreamZero
  "model|GEAR-Dreams/DreamZero-DROID|"
  "model|GEAR-Dreams/DreamZero-AgiBot|"

  # Fast-WAM
  "model|yuanty/fastwam|"

  # DiT4DiT
  "model|TeliMa/DiT4DiT|"
  "model|TeliMa/dit4dit_robocasa_gr1|"

  # RynnVLA-002 / WorldVLA
  "model|Alibaba-DAMO-Academy/RynnVLA-002|"

  # mimic-video
  "model|jonpai/mimic-video|"

  # LDA-1B
  "model|Wayer2/LDA-pretrain|"
  "model|Wayer2/LDA-robocasa|"

  # AIM
  "model|AUTMOEN999/AIM|"

  # X-WAM
  "model|sharinka0715/X-WAM-checkpoints|"
)

BACKBONE_REPOS=(
  # Wan/Cosmos video backbones used by multiple WAMs.
  "model|Wan-AI/Wan2.2-TI2V-5B|"
  "model|nvidia/Cosmos-Predict2-2B-Video2World|"
  "model|nvidia/Cosmos-Predict2.5-2B|diffusers/base/post-trained"

  # Motus/LDA language and vision backbones.
  "model|Qwen/Qwen3-VL-2B-Instruct|"
  "model|Qwen/Qwen3-VL-4B-Instruct|"
  "model|facebook/dinov3-vits16-pretrain-lvd1689m|"
)

VLA_BASELINE_REPOS=(
  # Optional non-WAM baselines mentioned for comparison.
  "model|openvla/openvla-7b|"
  "model|openvla/openvla-v01-7b|"
  "model|lerobot/smolvla_base|"
  "model|lerobot/smolvla_robotwin|"
  "model|x-square-robot/wall-oss-0.5|"
  "model|x-square-robot/wall-oss-flow|"
  "model|x-square-robot/wall-oss-fast|"
  "model|x-square-robot/wall-oss-flow-0.1|"
)

# Known gaps as of 2026-06-03:
#   - GigaWorld-Policy: official README lists pre-trained weights as not released.
#   - UWM: no official HF checkpoint repo confirmed; code exists on GitHub.
#   - UVA: README checkpoints are Google Drive links, not HF repos.
#   - WAV: no official HF checkpoint repo confirmed.
#   - WALL-WM: no official HF checkpoint repo confirmed.
#   - OpenPI pi0/pi0.5 official checkpoints are gs:// openpi assets, not HF.

FAILED=()

download_entry() {
  local entry="$1"
  local repo_type repo_id revision local_dir

  IFS='|' read -r repo_type repo_id revision <<<"$entry"
  local_dir="${SAVE_ROOT}/${repo_id}"

  mkdir -p "$local_dir"

  echo
  echo "==> Downloading ${repo_type}:${repo_id}"
  echo "    to ${local_dir}"
  if [[ -n "$revision" ]]; then
    echo "    revision ${revision}"
  fi

  local args=("$repo_id" "--local-dir" "$local_dir")
  if [[ "$repo_type" != "model" ]]; then
    args+=("--repo-type" "$repo_type")
  fi
  if [[ -n "$revision" ]]; then
    args+=("--revision" "$revision")
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${HF_CLI[@]}" "${args[@]}"
    printf '\n'
    return 0
  fi

  if ! "${HF_CLI[@]}" "${args[@]}"; then
    echo "Download failed: ${repo_id}" >&2
    FAILED+=("${repo_id}")
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit 1
    fi
  fi
}

prepare_lda_robocasa_layout() {
  local run_dir="${SAVE_ROOT}/Wayer2/LDA-robocasa"
  local src="${run_dir}/LDA-robocasa.pt"
  local dst_dir="${run_dir}/checkpoints"
  local dst="${dst_dir}/LDA-robocasa.pt"

  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  if [[ -f "$src" && ! -f "$dst" ]]; then
    mkdir -p "$dst_dir"
    mv "$src" "$dst"
    echo "Prepared LDA-robocasa layout: ${dst}"
  fi
}

main() {
  echo "Save root: ${SAVE_ROOT}"
  echo "HF endpoint: ${HF_ENDPOINT}"
  echo "Include backbones: ${INCLUDE_BACKBONES}"
  echo "Include VLA baselines: ${INCLUDE_VLA_BASELINES}"
  echo "Continue on error: ${CONTINUE_ON_ERROR}"
  echo "Dry run: ${DRY_RUN}"

  for entry in "${CORE_WAM_REPOS[@]}"; do
    download_entry "$entry"
  done

  if [[ "$INCLUDE_BACKBONES" == "1" ]]; then
    for entry in "${BACKBONE_REPOS[@]}"; do
      download_entry "$entry"
    done
  fi

  if [[ "$INCLUDE_VLA_BASELINES" == "1" ]]; then
    for entry in "${VLA_BASELINE_REPOS[@]}"; do
      download_entry "$entry"
    done
  fi

  prepare_lda_robocasa_layout

  echo
  if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo "Completed with failed downloads:"
    printf '  - %s\n' "${FAILED[@]}"
    exit 2
  fi

  echo "All selected downloads completed."
  echo "Files are saved under: ${SAVE_ROOT}"
}

main "$@"
