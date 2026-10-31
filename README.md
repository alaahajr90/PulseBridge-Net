# PulseBridge-Net — single-version, manuscript-aligned source staging

**Status: PRE-RELEASE / SOURCE-ALIGNED CANDIDATE. Not yet a verified reproduction of published results.**

This package intentionally displays **one canonical top-level folder per study stage**, rather than mixing historical v3/v4/v20 code as active implementations. The archived original multi-version package remains separate.

| Canonical folder | Selected implementation | Status |
|---|---|---|
| `preprocessing/` | original Data_QC_v2 + SHA-verified compact HDF5 extractor | **SOURCE PRESENT:** historical data rerun and split-file verification pending |
| `stage1/` | supplied seven-Conv3D contrastive encoder | structurally aligned; training not repeated |
| `stage2/` | **original Linux v4 MorphologyPreserving source** (verified SHA-256) | source identity verified; historical trained checkpoint identity still under audit |
| `stage3/` | supplied v20 with A2 variance coefficient set to 0 | code-level candidate; historical A2 run configs not verified |

There are **not** multiple active stage versions in this folder layout. Some filenames under `stage3/` contain historical version suffixes because the *selected* source imports/reuses those helper modules; they have not been deleted or misleadingly renamed.

## Important manuscript contracts

- 600 participants: fixed 60-subject independent cohort and 540 development subjects, with 10 matched subject-wise folds. Do not recreate test participants.
- Stage-1 input: `[B, 3, 128, 64, 64]` in extracted pixel scale, temporal encoder output `[B, 128, 64]`.
- Stage-2 original morphology-preserving v4: no Transformer; dilated Conv1D decoder with morphology-preserving terms Lm1, Lm2, Lms, plus reference waveform losses.
- Stage-3 selected A2: `V11_VARIANCE_WEIGHT=0.0` (this does **not** remove the waveform std loss from Stage 2). Matched-state D03 correction is conditional on paired segments.
- Full-development final fitting described in manuscript: Stage 2 for 118 epochs and Stage 3 for 13 epochs, with frozen checkpoints before independent evaluation; exact scripts/records are not supplied.


## Installation and environment documentation

**Research status:** this is a *pre-release, manuscript-aligned candidate*, not an
independently reproduced version of the published experiments. The original
workstation's exact historical dependency versions have **not** been supplied;
the pins below are an **illustrative, recommended installation profile** for
Python 3.10, not a claim about the training environment used to generate the
reported results.

- [Installation instructions (Linux, Conda, CPU/CUDA)](docs/INSTALLATION.md)
- [Dependency purpose and example exact versions](docs/DEPENDENCIES.md)
- [Original-environment capture instructions](docs/ENVIRONMENT_PROVENANCE.md)
- [Scientific/public-release readiness checklist](docs/RELEASE_READINESS.md)
- [`requirements-recommended.txt`](requirements-recommended.txt) — suggested
  exact versions for non-PyTorch packages; install PyTorch separately for your
  CUDA/CPU platform.
- [`environment.example.yml`](environment.example.yml) — example Conda environment
  (not the historical training environment).

Fast setup on a **supported Linux computer** with Python 3.10, after
installing the platform-appropriate PyTorch 2.5.1 wheel following the
[official PyTorch previous-version instructions](https://docs.pytorch.org/get-started/previous-versions/):

```bash
python -m pip install -r requirements-recommended.txt
python -m unittest discover -s tests -v
python scripts/capture_environment.py
python scripts/release_readiness_check.py
```

**Do not run full training or blind-test evaluation from this staging archive**:
the original frozen split/10-fold manifest files, historical Stage-2 v4 proof,
final model checkpoints and complete cross-stage integration still require verification. Historical source files contain absolute paths from the
original Linux workstation; they are not portable without explicit mapping.

Public visibility is **not yet recommended**. See the release-readiness report
for the precise scientific, technical, and rights-related blockers.

## Traceability

`provenance/` records the SHA-256 fingerprints of the source files used in this single-version staging package. Stage3's A2 edit relative to v20 was one loss-weight configuration change, **not** independently proven identical to the original recorded A2 training script. Original source archives and the earlier comprehensive audit must be retained in private storage.

## Check only (no dataset or training)

```bash
python -m unittest discover -s tests -v
```

This validates **static source contracts** and additionally performs Stage-2 synthetic PyTorch checks when PyTorch is available. It does not measure trained-model accuracy, source-to-checkpoint identity, or the published results. See `docs/SCIENTIFIC_MATCHING_AUDIT.md`.

Do not add a full-reproduction Code Availability claim until `stage2/` has historical v4 provenance and cross-stage hashes/contracts are verified, an appropriate license is approved, and the full pipeline passes tests.

## Stage-2 candidate correction (2026-10-30)

The `stage2/` folder now contains actual Python model, waveform loss and training
source reconstructed from the provided v3 code to match manuscript equations.
The original v4 source/run records remain unavailable; consequently this is
**source-level correspondence, not proof of historical experiment identity**.
See `stage2/README.md` and `docs/STAGE2_EQUATION_CROSSWALK.md`.

## Canonical HDF5 preprocessing and Data_QC_v2 (2026-10-30)

The exact original extractor and supplied Data_QC_v2 quality-control source
are now included in [`preprocessing/`](preprocessing/). The extractor SHA-256
was checked against the archived run contract, and the QC report checksums
matched the snapshot log. See [`docs/QC_V2_REPORT.md`](docs/QC_V2_REPORT.md).

No participant-level CSV, individual clinical outcomes, videos, raw PPG,
or generated HDF5 are included. The original frozen subject split was not
found by the supplied QC audit; historical Stage-2 v4/source checkpoint
matching, A2 contracts, end-to-end evaluation and licensing remain open.
