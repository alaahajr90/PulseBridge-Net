#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p results/logs results/preflight
exec 9>results/.stage2_training.lock
flock -n 9 || { echo "Stage-2 already running"; exit 2; }
echo "PulseBridge-Net Stage-2: manuscript-aligned SOURCE CANDIDATE, NOT historical verified v4"
python -u inspect_stage2.py 2>&1 | tee results/logs/preflight_latest.log
python -u train_stage2_cv.py --resume
