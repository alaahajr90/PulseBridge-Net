# Public-release readiness — PulseBridge-Net (31 Oct 2026)

**Scientific status:** suitable for supervisor source review; **not yet a
proven end-to-end reproduction**. Private repository is recommended until
remaining checkpoint/provenance, license and privacy gates are cleared.

| Gate | Current evidence | Status |
|---|---|---|
| Stage1 source | original supplied SSL source | present |
| Stage2 v4 source | original Linux v4; four core SHA-256 match workstation report | **source verified** |
| Stage2 final model & caches | full-dev checkpoint SHA and source-to-cache crosscheck | pending |
| Stage3 A2 code | v20-derived variance-off candidate | original training-source equivalence pending |
| Stage3 A2 final contract | Linux audit: 13 epochs, LR=1.5e-4, WD=2e-4, variance=0 | settings verified |
| Preprocessing | original compact HDF5 + Data_QC_v2 | source verified |
| Frozen split | Linux audit SHA matches Stage3 config; 540/60 + 10 valid folds | **verified** |
| Python CPU CI | data-free PyTorch source/model tests | basic smoke only, not scientific reproduction |
| Environment | original workstation current Python3.9.23/PyTorch2.5.1/CUDA11.8 | runtime snapshot, not historical lock |
| Rights/privacy | no open-source license yet; history not comprehensively audited | pending |

**Important:** the original Stage2 package's embedded PACKAGE_MANIFEST.json has
one inconsistent SHA for `inspect_stage2.py` (see source audit). Original absolute
Linux paths remain in verified code; this code requires the archived workstation
layout for full training. Do not modify historical code to 'fix' a checksum claim.

**Decision:** Keep Private for a *complete publication-reproducibility claim*.
Public availability as an openly labelled **source preview** is a distinct
author/editor decision after privacy and rights review; do not claim full
reproduction unless verified. No independent test rerun is necessary.
