#!/usr/bin/env bash

set -euo pipefail
set -x

PYTHON=${PYTHON:-python}
PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-"https://download.pytorch.org/whl/cu126"}
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-1}
INSTALL_POSTTRAIN=${INSTALL_POSTTRAIN:-1}

"${PYTHON}" - <<'PY'
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        "LingBot-VA environment should use Python 3.10.x "
        "(recommended: 3.10.16). Activate the conda env first or set PYTHON=..."
    )
PY

"${PYTHON}" -m pip install --upgrade pip setuptools wheel packaging ninja

# Keep LeRobot out of the resolver while installing the official LingBot-VA
# torch stack. LeRobot 0.3.3 declares torch<2.8, but this project uses torch 2.9.
"${PYTHON}" -m pip uninstall -y lerobot || true

"${PYTHON}" -m pip install --upgrade \
    "torch==2.9.0" \
    "torchvision==0.24.0" \
    "torchaudio==2.9.0" \
    --index-url "${PYTORCH_INDEX_URL}"

"${PYTHON}" -m pip install --upgrade \
    "websockets" \
    "einops" \
    "diffusers==0.36.0" \
    "transformers==4.55.2" \
    "accelerate>=0.31.0" \
    "huggingface-hub>=0.34.0,<1.0" \
    "msgpack" \
    "opencv-python" \
    "matplotlib" \
    "ftfy" \
    "easydict" \
    "numpy==1.26.4" \
    "tqdm" \
    "imageio[ffmpeg]" \
    "safetensors" \
    "Pillow" \
    "scipy" \
    "wandb"

if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then
    "${PYTHON}" -m pip install "flash-attn" --no-build-isolation
fi

if [ "${INSTALL_POSTTRAIN}" = "1" ]; then
    PYTHON="${PYTHON}" bash script/install_posttrain_deps.sh
fi

"${PYTHON}" - <<'PY'
import sys
import torch
import torchvision
import diffusers
import transformers
import accelerate
import huggingface_hub

print("Python:", sys.executable)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("huggingface_hub:", huggingface_hub.__version__)

try:
    import flash_attn
    print("flash_attn: installed")
except ImportError:
    print("flash_attn: not installed")

try:
    import lerobot
    print("lerobot:", getattr(lerobot, "__version__", "unknown"))
except ImportError:
    print("lerobot: not installed")
PY

cat <<'EOF'
LingBot-VA environment installation finished.

Example:
  PYTHON=$(which python) NGPU=1 bash script/run_va_contrastive_align.sh

If `pip check` reports that lerobot==0.3.3 requires torch<2.8, ignore that
metadata warning for this project. LingBot-VA follows the official torch 2.9
runtime and installs LeRobot with --no-deps for dataset APIs.
EOF
