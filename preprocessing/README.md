# Preprocessing and quality control

This directory contains the **exact original compact HDF5 extractor source**
`extract_mcd_rppg_clips_compact.py` (verified by its SHA-256) and the supplied
`Data_QC_v2` pipeline under `qcv2/`.

`qcv2/` includes the original QC algorithms and stage-specific inclusion
policies. Its path configuration was adapted to environment variables and
its full repair runner is protected against accidental invocation.

The original extraction format uses 128-frame clips, 64×64 face ROI and
stride 8. Stage-2 supervision uses authoritative provided synchronized PPG;
QC does **not** reconstruct those targets from raw PPG.

No raw recordings, participant-level manifests, physiological labels,
generated HDF5, or model checkpoints are redistributed here.
See [QC audit](../docs/QC_V2_REPORT.md).

**Caution:** QC verification of an original 600-person roster is not the
same as verification of the historically frozen 540/60 cohort split.
