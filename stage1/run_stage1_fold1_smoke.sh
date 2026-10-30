#!/usr/bin/env bash
set -euo pipefail
cd /home/hajr/mcd_rppg/code_HR_RR_model/PulseBridge_Net/Stage_1_PulseRepresentation
export PYTHONHASHSEED=42

python -u inspect_stage1_data.py

echo
echo "This command performs a REAL Fold-1 training run (up to normal early stopping)."
echo "It is optional. The normal full run is: bash run_stage1.sh"
python -u train_stage1_cv.py --fold 1 --resume
