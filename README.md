# PulseBridge-Net — single-version, manuscript-aligned source staging

**Status: PRE-RELEASE / SOURCE-ALIGNED CANDIDATE. Not yet a verified reproduction of published results.**

This package intentionally displays **one canonical top-level folder per study stage**, rather than mixing historical v3/v4/v20 code as active implementations. The archived original multi-version package remains separate.

| Canonical folder | Selected implementation | Status |
|---|---|---|
| `preprocessing/` | original Data_QC_v2 + SHA-verified compact HDF5 extractor | source supplied; frozen split hashes separately verified on Linux |
| `stage1/` | supplied seven-Conv3D contrastive encoder | structurally aligned; training not repeated |
| `stage2/` | **original Linux v4 MorphologyPreserving source** (verified SHA-256) | source identity verified; historical trained checkpoint identity still under audit |
| `stage3/` | authenticated original Linux A2_NoVarianceLoss source (13/13 SHA-256) | core source verified; final-13-epoch runner and checkpoint lineage remain open |

There are **not** multiple active stage versions in this folder layout. Some filenames under `stage3/` contain historical version suffixes because the *selected* source imports/reuses those helper modules; they have not been deleted or misleadingly renamed.

## Important manuscript contracts

- 600 participants: fixed 60-subject independent cohort and 540 development subjects, with 10 matched subject-wise folds. Do not recreate test participants.
- Stage-1 input: `[B, 3, 128, 64, 64]` in extracted pixel scale, temporal encoder output `[B, 128, 64]`.
- Stage-2 original morphology-preserving v4: no Transformer; dilated Conv1D decoder with morphology-preserving terms Lm1, Lm2, Lms, plus reference waveform losses.
- Stage-3 selected A2: `V11_VARIANCE_WEIGHT=0.0` (this does **not** remove the waveform std loss from Stage 2). Matched-state D03 correction is conditional on paired segments.
- Full-development final fitting described in manuscript: Stage 2 for 118 epochs and Stage 3 for 13 epochs. Original A2 contract confirms 13 epochs and source hashes; the separately fingerprinted final-development trainer file was not located.


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
source protocol SHA-256 and original Stage-2 v4 source are verified, but original split JSONs are not distributed;
final checkpoint/cache lineage, final Stage-3 runner and end-to-end integration still require verification. Historical source files contain absolute paths from the
original Linux workstation; they are not portable without explicit mapping.

Public visibility is **not yet recommended**. See the release-readiness report
for the precise scientific, technical, and rights-related blockers.

## Traceability

`provenance/` records the SHA-256 fingerprints of the source files used in this single-version staging package. The selected original Stage3 A2 core source is now byte-identical for all 13 files listed in the original A2 contract. The separate final-development trainer's SHA has not yet been traced to a source file. Original source archives and the earlier comprehensive audit must be retained in private storage.

## Check only (no dataset or training)

```bash
python -m unittest discover -s tests -v
```

This validates **static source contracts** and additionally performs Stage-2 synthetic PyTorch checks when PyTorch is available. It does not measure trained-model accuracy, source-to-checkpoint identity, or the published results. See `docs/SCIENTIFIC_MATCHING_AUDIT.md`.

Do not add a **full-reproduction** Code Availability claim until cross-stage checkpoint/cache/inference provenance is reconciled, an appropriate license and privacy clearance are approved, and the complete pipeline is verified. Stage-2 v4 and Stage-3 A2 original *source code* have now been authenticated.

## Authenticated original Stage-2 v4 and Stage-3 A2 sources (2026-10-31)

`stage2/` is genuine Linux v4 morphology-preserving source, replacing the prior
v3-derived candidate. Four main source fingerprints match the training machine.
See [Stage-2 source audit](docs/STAGE2_V4_SOURCE_AUDIT.md).

`stage3/` contains the **13 original selected A2 Python files**, each matching
the SHA-256 in the final-development A2 training contract. The previous
Stage3 `config_stage3.py` differed by three explanatory comments only; it is
now restored byte-for-byte to the historical source.
See [Stage-3 A2 audit](docs/STAGE3_A2_ORIGINAL_SOURCE_AUDIT.md).
The independent 13-epoch final trainer file remains unlocated, despite its
hash being recorded in the contract. These findings are source-authentication,
not numerical reproduction of published results.

## Canonical HDF5 preprocessing and Data_QC_v2 (2026-10-30)

The exact original extractor and supplied Data_QC_v2 quality-control source
are now included in [`preprocessing/`](preprocessing/). The extractor SHA-256
was checked against the archived run contract, and the QC report checksums
matched the snapshot log. See [`docs/QC_V2_REPORT.md`](docs/QC_V2_REPORT.md).

No participant-level CSV, individual clinical outcomes, videos, raw PPG,
or generated HDF5 are included. The original subject protocol hashes were
verified directly on the original Linux workstation; private split JSONs
are not included. Final checkpoint lineage, the separate A2 final runner,
full end-to-end evaluation, and license approval remain open.
