# Remaining verification items for public release

**Resolved at source level**

- `Data_QC_v2` and authentic compact HDF5 extraction source, with matching archived
  extraction hash. Original per-record QC outputs are not redistributed.
- Original Linux Stage-2 v4 morphology-preserving Python source; exact `model`,
  `losses`, `config`, `train` hashes match 31 October 2026 workstation audit.
- Frozen 540-development/60-independent subject split and 10 folds have been
  verified on the original Linux workstation against Stage3 embedded SHA-256;
  source partition files are not redistributed in this ZIP.
- Stage3 A2 final-fit contract confirms 13 epochs, LR 1.5e-4, WD 2e-4,
  seed 42, variance-weight 0.

**Still require evidence / author decision**

1. Correlate Stage2 10-fold **checkpoint SHA-256** hashes and 118-epoch
   full-development final checkpoint with the training contracts and Stage3
   rPPG/F7 caches (including original /255 vs raw-scale audit paths).
2. Correlate Stage3 selected A2 **original trained source and final-fit
   checkpoint hashes**, per-fold contracts, and downstream inference scripts
   against the actual 10-fold and final results. The GitHub Stage3 source is
   currently a v20-derived A2 candidate, not a hash-matched historical original.
3. Verify that the original final-development inference and metric-generation
   scripts are included or clearly document the remaining gaps.
4. Approve license and contributor rights; scan the entire Git history for
   sensitive files, private measurements, credentials and recordings.
5. Pin an actual historical training environment if archived; 31 Oct 2026 Linux
   versions are a **current workstation snapshot**, not necessarily historical.

No repeated blind-test evaluation or split regeneration is required for these
source/provenance checks. The previous v4 README refers to an earlier-phase
blind-test status and must not override the manuscript's final frozen protocol.
