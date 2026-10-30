# Stage 2 — morphology-preserving manuscript-aligned **derived candidate**

**Source:** `code stage2_new.zip` (`FINAL_v3`), modified only as documented to
implement the v4 loss terms in manuscript Eqs. (11)–(17), expose F7, and retain
original architecture and participant-level split protocol.

This is **not authenticated historical Stage-2 v4 source**. It must not be
used to claim reproduction of the paper's reported Stage-2 `r`, Stage-3 BP
results, or final checkpoints without training/provenance verification.

## Changes from the archived v3 source

- Eq. (11): per-sample population-standard-deviation normalization of temporal derivatives.
- Eq. (12): `Lm1 = SmoothL1(Z(Δp̂),Z(Δp))`, coefficient 0.15.
- Eq. (13): `Lm2 = SmoothL1(Z(Δ²p̂),Z(Δ²p))`, coefficient 0.35.
- Eq. (14): `Lms = mean_{lag∈{1,2,4}} SmoothL1(Z(Δ_lag p̂),Z(Δ_lag p))`, coefficient 0.20.
- Eq. (15): total waveform loss preserves existing 3.00/0.75/0.50/0.75/0.05/0.25 weights.
- Eqs. (16)–(17): retains weak auxiliary loss 0.15 × mean(three heads).
- `Stage2RPPGModel.forward()` additionally exposes `f7: [B,T,64]`, without
  changing existing keys or checkpoint parameter names.
- Training contract and validation outputs record the new loss terms.

The original v3 source hashes are preserved in `../provenance/STAGE2_V3_SOURCE_SHA256.json`.

## Experimental protocol carried from the original code

- `[B,3,128,64,64]` unscaled RGB input (`SCALE_TO_01=False`).
- All seven Stage-1 Conv3D layers fine-tuned, with early/late LR 1e-5/5e-5;
  encoder BN frozen, decoder LR 2e-4.
- Dilations 1,2,4,8; 64-channel attention-free temporal decoder; three auxiliary heads.
- Batch preflight 16→8→4→2; 10 subject-exclusive development folds;
  validation-selected checkpoints, max 120 epochs and patience 18.
- Frozen 60-subject test assignment is not rebuilt or accessed by audit tests.

**Data dependency:** original authoritative HDF5 ROI/PPG clips, Stage-1 frozen
splits, fold-specific Stage-1 checkpoints and QC manifest with recorded hashes.
The package does not contain private recordings or checkpoints.

## Tests

From repository root: `python -m unittest discover -s tests -v`.

For actual experiments, first compare this code to the original v4 source, and
run only on the development cohort using the frozen manifests. Do not launch
blind-test inference for version matching.
