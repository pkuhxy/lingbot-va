#!/usr/bin/env bash

set -euo pipefail
set -x

python -m pip install --upgrade "accelerate>=0.31.0"

# Repair environments where a later pip command upgraded datasets to 5.x.
python -m pip install --upgrade --force-reinstall "datasets>=2.19.0,<=3.6.0"

python -m pip install --upgrade -r requirements-posttrain.txt

# Do not let pip resolve LeRobot's torch<2.8 metadata; LingBot-VA uses torch 2.9.
python -m pip install --upgrade --no-deps "lerobot==0.3.3"
