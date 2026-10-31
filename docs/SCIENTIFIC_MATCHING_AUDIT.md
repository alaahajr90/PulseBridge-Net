# PulseBridge-Net — updated source/manuscript scientific audit

Reference manuscript: `main(20261030-154034).pdf` (42 pages).
This is a source-level matching audit, **not** a repetition of data training,
benchmark outcomes, or one-time independent evaluation.

| Subject | Source evidence | Status |
|---|---|---|
| Data QC | authenticated original extractor and Data_QC_v2 modules | source received; real-data rerun not performed |
| Frozen protocol | Linux original split/fold hashes match Stage3 expected hashes; 540/60, 10x 486/54 | independently checked from workstation report |
| Stage1 | original 7-Conv3D SSL source; no raw medical targets | structure inspected |
| Stage2 | **genuine** original Linux v4 source | principal Python hashes match workstation audit |
| Stage2 morphology | `normalized_d1_shape`, `normalized_d2_shape`, multi-scale increments, Eq. (11–17) | implemented in original v4; synthetic unit tests |
| Stage2 output F7 | `forward_late_features(x)[-1]` | verified source interface; not an `out['f7']` key |
| Stage2 raw RGB scale | `SCALE_TO_01=False` | verified training config |
| Historical Stage2 cache generation | earlier OOF cache path `/255.0` observed in separate script | potential discrepancy; requires archive-level traceability |
| Stage2 v4 final 118 epochs / batch16 | manuscript | contract/weights not independently verified here |
| Stage3 selected A2 | `V11_VARIANCE_WEIGHT=0` candidate source; original A2 final contract recorded `epochs=13`, `lr=1.5e-4` | contract settings verified, full training-source identity pending |
| Full reproduction of HR/rPPG/BP/stiffness results | datasets/checkpoints/logs not distributed | not performed |

Do not re-evaluate independent test subjects to reconcile release source files.
See `STAGE2_V4_SOURCE_AUDIT.md` and `MISSING_SOURCES_AND_RELEASE_BLOCKERS.md`.
