# PulseBridge-Net — scientific source-to-manuscript audit

**Audit basis:** 42-page manuscript `main(20261029-164645).pdf`, uploaded
Stage-1, Stage-2 v3, Stage-2 v4 audit, Stage-3 v20 and clip-extractor archives.
This review compares sources, configuration semantics and code contracts; it
does not replay training or verify published MAE from actual MCD-rPPG data.

| Manuscript requirement | Located source | Finding |
| --- | --- | --- |
| Stage-1 7 Conv3D architecture; output 128x64 (§3.3) | `stage1/model_stage1.py` | **Source aligned** (structural). |
| Stage-1 τ=.07; margin=.30; temporal λ=.25 | `stage1/config_stage1.py`, `model_stage1.py` | **Aligned by inspection**. |
| Stage-1 128x64x64, not scaled to [0,1] | `stage1/config_stage1.py`, `data_stage1.py` | **Aligned**: SCALE_TO_01=False. |
| Frozen roster 60 test/540 development, 10 folds | `stage1/split_protocol.py` and Stage-1 config | **Contract present; frozen JSON not provided**; hash validation needed. |
| Stage-2 attention-free 4-block dilated TCN | `stage2_v3_original/model_stage2.py` | **Architecture appears aligned**, but manuscript v4 source unavailable. |
| Stage-2 3 morphology-preserving penalties Lm1/Lm2/Lms, Eq.(12-17) | `stage2_v3_original/losses_stage2.py` | **MISMATCH**: supplied v3 objective lacks these penalties; original v4 must be provided. |
| Stage-2 freeze BN; 1e-5/5e-5/2e-4 LRs | `stage2_v3_original/model_stage2.py`, `config_stage2.py` | v3 **aligned**; v4 implementation not verified. |
| Stage-2 v4 10-fold validation caches, F7 | `stage2_v4_oof_audit/` | **Audit only**, not source; requires preexisting v4 checkpoints. |
| ROI raw RGB scale not divided by 255 (§3.2) | `stage2_v4_oof_audit/build_oof_cache_v4.py` | **SCALE DISCREPANCY**: cache builder divides by 255; audit with original v4 before running. |
| Stage-3 frozen rPPG and [N,128,64] F7, camera features | `stage3_v20_reference/data_stage3.py`, `model_stage3.py` | **Interfaces present**; no original v4 source/checkpoints/caches. |
| Stage-3 4 amplitude + 16 physiology descriptors | `stage3_v20_reference/config_stage3.py`, `data_stage3.py` | **Source aligned** by inspection. |
| Stage-3 coupled SBP correction, D03 six-vector, 0.5 anticorrection (§3.5) | `stage3_v20_reference/model_stage3.py`, `paired_loss_v20.py` | **Source aligned** by inspection; paired infer requires matched states. |
| Stage-3 selected A2 λvar=0, λpair=0.20 (§3.6) | `stage3_v20_reference/config_stage3.py` | **MISMATCH in original v20** (variance=0.05). `stage3_a2_selected/` contains the documented one-line adjustment, **not independently confirmed as the historical A2 run script**. |
| Stage-3 max180, patience28, batch8, LR1.5e-4 | `stage3_v20_reference/config_stage3.py`, `train_stage3_cv.py` | **Config mostly aligned**; hardware batch is selected at run time, logs/contracts required. |
| Final Stage-2 118 epochs, Stage-3 13 epochs; independent holdout evaluation (§4.2) | Provided archives | **MISSING exact final-fit scripts/checkpoints, frozen scalers and holdout inference code**. |
| Fixed 60-subject independent evaluation | Manuscript §5.8 | **Reported in manuscript**; independent predictions/label-access evidence not in ZIP. |
| Data_QC_v2 authoritative HDF5 manifests | legacy `.npz` extractor only | **MISMATCH**: authoritative Data_QC_v2 code not provided. |

## Critical interpretation

- The 60-subject test is described in the latest manuscript as evaluated **once after protocol freezing**; this audit never accesses it and makes no contrary claim.
- The original Stage-3 v20 code remains intact in `stage3_v20_reference`. It is not relabeled as A2.
- The selected A2 source is a transparent, minimal edit of the reference code, pending comparison against **actual archived A2 training contracts/checkpoints**.
- The `stage2_v3_original` source is deliberately not promoted to v4. A visually similar decoder is insufficient evidence of the published morphology-preserving loss.
- Stage-3 labels `rigidity` are dataset-defined stiffness values; paper correctly does not assume PWV units.
- All data paths in source archives point to the original research workstation and are *not* portable defaults.

## Validation status

- ZIP integrity / Python syntax / source fingerprints: audited in `provenance/`.
- Synthetic architecture tests: see `tests/` and `docs/TEST_REPORT.md`.
- Actual dataset training, 10-fold CV, independent predictions and MAEs: **NOT performed in this audit**.


## Addendum (2026-10-30)

A derived Stage-2 paper-aligned source candidate has now been added in `stage2/`.
The earlier "MISSING" statements refer to **historical original v4 source**, not
to availability of a newly reconstructed implementation. The original Stage2 v4
source-to-checkpoint provenance is still not verified. See `STAGE2_EQUATION_CROSSWALK.md`.
