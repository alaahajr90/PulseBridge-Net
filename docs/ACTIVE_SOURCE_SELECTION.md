# Canonical source selection and manuscript correspondence

- `preprocessing/`: original compact HDF5 extractor and Data_QC_v2 source (source-level verified).
- `stage1/`: supplied SSL Conv3D encoder source.
- `stage2/`: **authenticated original Stage-2 v4 MorphologyPreserving source from
  the Linux training workstation**. Principal four hashes matched the workstation
  audit. The previous v3-derived manuscript-aligned candidate is no longer active.
- `stage3/`: v20-derived A2 candidate with `V11_VARIANCE_WEIGHT=0`; selected
  final-development training-contract properties have been verified from an
  original Linux audit, but complete source/weight/run identity remains to be checked.

Frozen subject protocol: 540 development / 60 independent, ten disjoint folds
(486/54). Original Stage1 protocol SHA-256 hashes matched the Stage3 source
configuration on the Linux machine. Do not export raw subject IDs by default.

Published numerical metrics have **not** been reproduced by these data-free tests.
