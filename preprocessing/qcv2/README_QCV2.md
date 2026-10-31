# PulseBridge-Net Data_QC_v2 + True Repair

This package is grounded in the actual compact extractor source code.

## Critical correction

The compact extractor does NOT create synchronization from raw `.PW`.
It reads the already-provided `ppg_sync/<sid>_<camera>_<state>.txt`
and copies its amplitude and delta-t into HDF5.

Therefore:
- source `ppg_sync` is never reconstructed or overwritten here;
- BAD/REVIEW sync is a Stage-2 supervision issue, not a universal recording failure.

## Decode repair

The original extractor used:
`n_usable = min(n_meta, n_sync)`

but sequential video decoding can stop one or more frames earlier.

For only the extraction-error queue, this package determines the actual
sequentially decodable video length and uses:

`n_usable = min(n_meta, n_sync, n_actual_decodable)`

It then reuses the ORIGINAL extractor module and ORIGINAL ROIExtractor.
No frame is duplicated/interpolated. No raw source is modified.

## Stage-specific inclusion

Stage 1:
- structural HDF5
- acceptable ROI
- synchronization NOT used as a gate

Stage 2:
- structural HDF5
- acceptable ROI
- GOOD authoritative `ppg_sync`

Stage 3:
- structural HDF5
- acceptable ROI
- finite SBP, DBP, rigidity
- source synchronization NOT used as a gate because Stage 3 consumes predicted rPPG

## Frozen subject protocol

QC never regenerates subject splits. The original 600-subject roster remains the
population from which the fixed blind/dev protocol is defined.

Run:
`bash run_all_qcv2_and_repair.sh`

## Publication source package integration (2026-10-30)
The scripts above come from the author's supplied `Data_QC_v2.zip`. All core
QC calculations and thresholds are retained; `config_qcv2.py` now takes
environment paths, and `run_all_qcv2_and_repair.sh` requires explicit
permission before its destructive-to-derived-HDF5 *repair* step.

```bash
export PULSEBRIDGE_DATA_ROOT=/path/to/original/MCD-rPPG
export PULSEBRIDGE_CLIP_ROOT=/path/to/derived/clips
export PULSEBRIDGE_LANDMARKS=/path/to/shape_predictor_68_face_landmarks.dat
export PULSEBRIDGE_STAGE1_PROTOCOL=/path/to/ORIGINAL/frozen/protocol
export PULSEBRIDGE_ALLOW_QC_REPAIR=YES
bash preprocessing/qcv2/run_all_qcv2_and_repair.sh
```

**Do not run** this on the 60-subject independent evaluation pipeline to
regenerate data or create alternative split assignments. Make backups of
derived recording.h5 files before running the repair stage. This workflow
is provided as source provenance, not a command to repeat the completed experiment.

The supplied QC evidence includes participant-level records and labels;
these files are **not included in this public-safe source package**.
Summary counts and cryptographic hashes are recorded in `docs/QC_V2_REPORT.md`.

The extractor is `preprocessing/extract_mcd_rppg_clips_compact.py`
and its SHA-256 matches the frozen original extractor hash.
