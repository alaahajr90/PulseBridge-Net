# PulseBridge-Net — source/manuscript scientific matching audit

Reference manuscript: `main(20261030-154034).pdf` (42 pages).
This audit compares source evidence with manuscript statements; it does **not**
repeat model training, independent testing, or physiological metric computation.

| Component | Evidence | Status |
|---|---|---|
| Input preprocessing | authenticated compact extractor / Data_QC_v2 | source supplied; data rerun not done |
| Split / ten folds | source config SHA matches original Linux split & folds; 540/60, 10 × 486/54 | verified on workstation |
| Stage 1 | original supplied Conv3D self-supervised source | structure inspected |
| Stage 2 v4 | original Linux model, losses, config, CV train SHA-256 | source verified |
| Stage 2 morphology | normalized 1st/2nd differences and lags 1/2/4 | implemented in original v4; synthetic tests |
| Stage 2 F7 | `forward_late_features(x)[-1]` | source interface checked |
| Stage 2 RGB scale | `SCALE_TO_01=False` | config verified; separate old cache /255 ambiguity open |
| Stage 2 final 118 epochs | manuscript; checkpoint hashes/caches incomplete | lineage pending |
| Stage 3 selected A2 | original 13/13 Python source SHA-256 equal training contract | **source verified** |
| Final A2 optimization | contract: epochs13, batch8, LR1.5e-4, WD2e-4, variance0, pair0.2 | settings verified |
| Final trainer separately fingerprinted | SHA-256 exists; no matching file located | outstanding |
| Full clinical accuracy reproduction | no dataset/checkpoint evaluation in this audit | **not performed** |

For Stage2 and Stage3 detailed evidence see `STAGE2_V4_SOURCE_AUDIT.md`
and `STAGE3_A2_ORIGINAL_SOURCE_AUDIT.md`. Do not rerun the independent test
set for provenance validation.
