# Canonical PulseBridge-Net study implementation

The top-level research implementation selects **one code path per stage**:

- `preprocessing/`: original compact HDF5 extractor and Data_QC_v2 sources.
- `stage1/`: original supplied seven-Conv3D self-supervised encoder source.
- `stage2/`: authentic morphology-preserving **v4** source recovered from the
  original Linux training workstation; four principal file hashes and data-free
  tests verified. See `STAGE2_V4_SOURCE_AUDIT.md`.
- `stage3/`: authentic **A2_NoVarianceLoss** source. All **13/13** filenames
  listed in the A2 final-development training contract match original Linux
  SHA-256 fingerprints. See `STAGE3_A2_ORIGINAL_SOURCE_AUDIT.md`.

**Training protocol**: original Linux subject split SHA-256 verified; 540
subjects for development, 60 independent subjects, 10 subject-exclusive folds
(486/54), no development/test overlap. Subject identifiers and clinical files
are not distributed in this source repository.

**Not established by these audits:** complete end-to-end numerical reproduction;
original standalone 13-epoch final Stage-3 trainer source (its fingerprint is
recorded, but the script has not yet been located); complete Stage-2 final
checkpoint/cache lineage; privacy/license clearance for public distribution.
