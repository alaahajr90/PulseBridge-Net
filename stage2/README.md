# Stage 2 — authenticated original Linux v4 MorphologyPreserving source

**Canonical source:** Author-supplied `Stage_2_rPPG_FineTuning_v4_MorphologyPreserving` archive received 31 October 2026.
The 13 Python source files in this folder are **byte-identical** to the corresponding
files supplied from the Linux training workstation. Four principal hashes match the
workstation's source-audit report. See `../provenance/STAGE2_V4_ORIGINAL_SOURCE_SHA256.json`.

The training workstation is Linux (`/home/hajr/...`), *not* the MacBook used for GitHub commits.
Historical absolute workstation paths in `config_stage2.py` are retained deliberately to preserve
source identity. They must be configured for another training machine; this repository's GitHub CI
only performs data-free tests and does not attempt to train.

## Manuscript match

- ROI input `[B,3,128,64,64]`, `SCALE_TO_01=False`, Stage-1 Conv3D initialization;
  encoder Conv3D fine-tuning with frozen BatchNorm, TCN dilations `(1,2,4,8)`;
  three auxiliary waveform heads.
- The waveform loss contains Pearson, waveform, derivative, spectral, pulse-band,
  waveform-standard-deviation terms and the original auxiliary-head term.
- Authentic v4 adds `normalized_d1_shape_per_sample` (0.15),
  `normalized_d2_shape_per_sample` (0.35), and `multiscale_shape_per_sample` (0.20,
  lags 1/2/4), as described in manuscript Eqs. (11–17).
- Original `Stage2RPPGModel.forward()` yields `rppg`, `aux_rppg` and `all_rppg`.
  The final 64-dimensional `F7` sequence is obtained via
  `model.forward_late_features(x)[-1]`, **not** via `forward()['f7']`.
  Model parameters and tensor shapes are unchanged.
- `train_stage2_cv.py` implements development 10-fold training. The original
  `run_stage2_v4_morph.sh` is **historical CV-only** (`--skip-final`). A final
  development fit is described in the manuscript, but its archived run contract
  and exact checkpoint provenance have not been independently checked here.

## Historical-source limitations

- Authenticated source-code identity does not itself prove the SHA-256 of trained
  Stage-2 v4 fold/final checkpoints or reproduce published metrics.
- The original `PACKAGE_MANIFEST.json` has an inconsistent hash for
  `inspect_stage2.py`; this is explicitly recorded in the provenance report.
- One older v4 README describes a blind-test status from an earlier experiment.
  It is not used to revise the frozen evaluation protocol in the manuscript.
- Some code needs original HDF5 data, QC manifests and frozen Stage-1 checkpoints;
  none are distributed here. This source must not be used to re-tune on the
  independent 60-subject test cohort.

See `../docs/STAGE2_V4_SOURCE_AUDIT.md` for code-to-paper details.
