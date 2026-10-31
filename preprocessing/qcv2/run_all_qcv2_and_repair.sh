#!/usr/bin/env bash
set -euo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${PULSEBRIDGE_ALLOW_QC_REPAIR:-}" != "YES" ]]; then
  echo "Refusing to run derived-HDF5 repair unless PULSEBRIDGE_ALLOW_QC_REPAIR=YES is set." >&2
  echo "Review source data paths, generated HDF5 backups and source code first." >&2
  exit 2
fi
mkdir -p results logs backups _repair_tmp

LOG="logs/qcv2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "===================================================================================================="
echo "PulseBridge-Net — DATA QC v2 + TRUE DECODE REPAIR"
echo "Grounded directly in extract_mcd_rppg_clips_compact.py"
echo "START: $(date)"
echo "===================================================================================================="

echo
echo "[0/8] Verify exact extractor contract..."
python -u 00_verify_source_contract.py

echo
echo "[1/8] Repair extraction decode failures using exact original ROI extractor..."
python -u 01_repair_decode_errors.py

echo
echo "[2/8] Full HDF5 / ROI / source-sync audit..."
python -u 02_full_audit_v2.py

echo
echo "[3/8] Clinical label integrity..."
python -u 03_label_integrity_v2.py

echo
echo "[4/8] Build separate Stage 1 / Stage 2 / Stage 3 manifests..."
python -u 04_build_stage_specific_manifests.py

echo
echo "[5/8] Register source ppg_sync anomalies without modifying them..."
python -u 05_sync_source_anomaly_register.py

echo
echo "[6/8] Verify original 600-subject roster and frozen protocol..."
python -u 06_protocol_roster_audit.py

echo
echo "[7/8] Stage-specific final preflight..."
python -u 07_final_preflight_v2.py

echo
echo "[8/8] Freeze QC v2 report hashes..."
python -u 08_freeze_qcv2_snapshot.py

rm -rf _repair_tmp/* 2>/dev/null || true

echo "===================================================================================================="
echo "FINISHED: $(date)"
echo "LOG: $LOG"
echo "FINAL: results/07_final_preflight_v2.json"
echo "STAGE 1: results/04_stage1_manifest.csv"
echo "STAGE 2: results/04_stage2_manifest.csv"
echo "STAGE 3: results/04_stage3_manifest.csv"
echo "===================================================================================================="
