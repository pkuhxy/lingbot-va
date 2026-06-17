#!/usr/bin/env bash

set -euo pipefail
set -x

PYTHON=${PYTHON:-python}

# Remove LeRobot while repairing dependencies so pip does not try to satisfy
# its stale torch<2.8 metadata against the LingBot-VA torch 2.9 stack.
"${PYTHON}" -m pip uninstall -y lerobot || true

"${PYTHON}" -m pip install --upgrade "accelerate>=0.31.0" "huggingface-hub>=0.34.0,<1.0"

# Repair environments where a later pip command upgraded datasets or hub too far.
"${PYTHON}" -m pip install --upgrade --force-reinstall "datasets>=2.19.0,<=3.6.0"
"${PYTHON}" -m pip install --upgrade --force-reinstall "huggingface-hub>=0.34.0,<1.0"

"${PYTHON}" -m pip install --upgrade -r requirements-posttrain.txt

# Do not let pip resolve LeRobot's torch<2.8 metadata; LingBot-VA uses torch 2.9.
"${PYTHON}" -m pip install --upgrade --no-deps "lerobot==0.3.3"

"${PYTHON}" - <<'PY'
import sys
import lerobot
import datasets
import huggingface_hub

print("Python:", sys.executable)
print("lerobot:", getattr(lerobot, "__version__", "unknown"))
print("datasets:", datasets.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
PY

cat <<'EOF'
Post-training dependencies installed.

Note: `pip check` may still report that lerobot==0.3.3 requires torch<2.8
and torchvision<0.23. This is expected for LingBot-VA: the official runtime
uses torch==2.9.0 / torchvision==0.24.0, and LeRobot is installed with
--no-deps only for its dataset APIs.

If a code path needs torchcodec, install a torch-2.9-compatible torchcodec
wheel separately. The latent post-training dataset path uses pyav/precomputed
latents and does not require torchcodec in the normal flow.
EOF
