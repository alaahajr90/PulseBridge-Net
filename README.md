# PulseBridge-Net — single-version, manuscript-aligned source staging

**Status: PRE-RELEASE / SOURCE-ALIGNED CANDIDATE. Not yet a verified reproduction of published results.**

This package intentionally displays **one canonical top-level folder per study stage**, rather than mixing historical v3/v4/v20 code as active implementations. The archived original multi-version package remains separate.

| Canonical folder | Selected implementation | Status |
|---|---|---|
| `preprocessing/` | original Data_QC_v2 | **BLOCKED:** original source not supplied |
| `stage1/` | supplied seven-Conv3D contrastive encoder | structurally aligned; training not repeated |
| `stage2/` | manuscript-aligned *derived* candidate from supplied v3 | **PRESENT:** equations/architecture mapped to paper; historical v4 checkpoint identity not verified |
| `stage3/` | supplied v20 with A2 variance coefficient set to 0 | code-level candidate; historical A2 run configs not verified |

There are **not** multiple active stage versions in this folder layout. Some filenames under `stage3/` contain historical version suffixes because the *selected* source imports/reuses those helper modules; they have not been deleted or misleadingly renamed.

## Important manuscript contracts

- 600 participants: fixed 60-subject independent cohort and 540 development subjects, with 10 matched subject-wise folds. Do not recreate test participants.
- Stage-1 input: `[B, 3, 128, 64, 64]` in extracted pixel scale, temporal encoder output `[B, 128, 64]`.
- Stage-2 morphology-preserving derived candidate: no Transformer; dilated Conv1D decoder with morphology-preserving terms Lm1, Lm2, Lms, plus reference waveform losses.
- Stage-3 selected A2: `V11_VARIANCE_WEIGHT=0.0` (this does **not** remove the waveform std loss from Stage 2). Matched-state D03 correction is conditional on paired segments.
- Full-development final fitting described in manuscript: Stage 2 for 118 epochs and Stage 3 for 13 epochs, with frozen checkpoints before independent evaluation; exact scripts/records are not supplied.

## Traceability

`provenance/` records the SHA-256 fingerprints of the source files used in this single-version staging package. Stage3's A2 edit relative to v20 was one loss-weight configuration change, **not** independently proven identical to the original recorded A2 training script. Original source archives and the earlier comprehensive audit must be retained in private storage.

## Check only (no dataset or training)

```bash
python -m unittest discover -s tests -v
```

This validates **static source contracts** and additionally performs Stage-2 synthetic PyTorch checks when PyTorch is available. It does not measure trained-model accuracy, source-to-checkpoint identity, or the published results. See `docs/SCIENTIFIC_MATCHING_AUDIT.md`.

Do not add a public Code Availability claim until `preprocessing/` and `stage2/` have their historical v4 provenance and cross-stage hashes/contracts are verified, an appropriate license is approved, and the full pipeline passes tests.

## Stage-2 candidate correction (2026-10-30)

The `stage2/` folder now contains actual Python model, waveform loss and training
source reconstructed from the provided v3 code to match manuscript equations.
The original v4 source/run records remain unavailable; consequently this is
**source-level correspondence, not proof of historical experiment identity**.
See `stage2/README.md` and `docs/STAGE2_EQUATION_CROSSWALK.md`.
