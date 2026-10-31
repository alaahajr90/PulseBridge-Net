# PulseBridge-Net — verified source handoff for supervisor review

**Source state as of 31 Oct 2026**

- Original Linux video preprocessing / Data_QC_v2 integrated.
- Stage 1 original author-supplied Conv3D source retained.
- Authentic Linux Stage 2 v4 morphology-preserving model, losses,
  configuration and CV trainer verified by SHA-256; original stage source is
  published in `stage2/`, not a reimplementation from v3.
- **Stage 3 selected A2: 13/13 core source files match the original final-fit
  A2 training contract by SHA-256.** The only previous difference was three
  explanatory comment lines in `stage3/config_stage3.py`. The authenticated
  original configuration is now restored byte-for-byte.
- Frozen original participant protocol hashes match recorded source hashes:
  540 development, 60 fixed independent, 10 disjoint development folds
  (486/54), zero development/test overlap.
- Historical A2 final-fit contract: seed42, epochs13, batch8, LR1.5e-4,
  WD2e-4, `variance_weight=0`, `pair_weight=0.2`.
- GitHub Actions synthetic PyTorch unit tests passed on previous release;
  run fresh CI on this commit before claiming the latest commit passed.

**Open scientific artifact provenance**

1. The distinct **final-training runner** SHA-256
   `f658fe2f8712606108322c709bbf2f04e0e0b01011b06910664de77437435616`
   is present in the contract but its source was **not located** in the
   searched experiment project. `train_stage3_cv.py` is the CV trainer, not
   proven to be the original final runner.
2. Match frozen full-development Stage2 checkpoint and Stage3 input caches,
   verify source-to-checkpoint-to-metrics traceability including historical
   raw-pixel scaling in all relevant pathways.
3. Complete license/rights approval, scan tracked files and **all Git history**
   for confidential data, and optionally locate historical environment locks.

**Appropriate external description:** authenticated Stage2-v4 and Stage3-A2
source under final experimental lineage verification. Not yet independently
reproduced published metrics or fully portable turnkey training. No training,
additional blind test or model retuning is needed solely for this documentation.
