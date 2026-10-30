#!/usr/bin/env bash
set -euo pipefail

STAGE1="/home/hajr/mcd_rppg/code_HR_RR_model/PulseBridge_Net/Stage_1_PulseRepresentation"
cd "$STAGE1"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p results/logs results/preflight

# Prevent two Stage-1 training jobs from writing the same checkpoints.
exec 9>results/.stage1_training.lock
if ! flock -n 9; then
  echo "ERROR: another Stage-1 process holds results/.stage1_training.lock"
  exit 2
fi

echo "===================================================================================================="
echo "PulseBridge-Net — Stage 1 FINAL"
echo "1) Frozen QC manifest + protocol + loader + GPU preflight"
echo "2) Subject-balanced 10-fold CV"
echo "3) Final development encoder"
echo "Blind test remains LOCKED."
echo "===================================================================================================="

python -u inspect_stage1_data.py 2>&1 | tee results/logs/preflight_latest.log

# --resume is safe: only checkpoints with an identical frozen training contract
# can be reused. On a fresh run it behaves exactly like normal training.
python -u train_stage1_cv.py --resume

echo "===================================================================================================="
echo "STAGE 1 FINISHED"
echo "Final encoder:"
echo "  results/final_model/stage1_encoder_for_stage2.pth"
echo "Blind test:"
echo "  LOCKED / NOT EVALUATED IN STAGE 1"
echo "===================================================================================================="
