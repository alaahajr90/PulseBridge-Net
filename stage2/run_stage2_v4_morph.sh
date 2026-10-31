#!/usr/bin/env bash
set -euo pipefail

STAGE2="/home/hajr/mcd_rppg/code_HR_RR_model/PulseBridge_Net/Stage_2_rPPG_FineTuning_v4_MorphologyPreserving"
cd "$STAGE2"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ulimit -n 65535

mkdir -p results/logs results/preflight
exec 9>results/.stage2_v4_morph_training.lock
if ! flock -n 9; then
  echo "ERROR: another Stage-2 v4 morphology process holds the lock"
  exit 2
fi

echo "===================================================================================================="
echo "PulseBridge-Net — Stage 2 v4 MorphologyPreserving"
echo "CONTROLLED DEVELOPMENT EXPERIMENT: preflight -> 10-fold CV ONLY -> v3 comparison"
echo "Original Stage-2 v3 architecture/data/protocol/optimizer/loss remain intact."
echo "ONLY added losses: normalized d1 + normalized d2 + multiscale temporal shape."
echo "Consumed 60-subject Blind is FORBIDDEN."
echo "===================================================================================================="

python -u inspect_stage2.py 2>&1 | tee results/logs/preflight_latest.log
python -u train_stage2_cv.py --resume --skip-final
python -u compare_stage2_v3_v4.py | tee results/logs/v3_vs_v4_latest.log

echo "===================================================================================================="
echo "STAGE 2 v4 MORPHOLOGY CV FINISHED"
echo "No final-development Stage2 model was fitted yet."
echo "No Blind evaluation was performed."
echo "NEXT SCIENTIFIC STEP: build v4 out-of-fold predictions and rerun the SAME morphology audit."
echo "===================================================================================================="
