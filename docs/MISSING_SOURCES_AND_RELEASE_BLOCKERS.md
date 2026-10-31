# Remaining documentation and publication-release gates

## Source-level verification completed

- Original compact HDF5 extraction and Data_QC_v2 sources authenticated;
  individual per-record clinical output data are not released.
- Original Stage2 v4 `model_stage2.py`, `losses_stage2.py`, `config_stage2.py`,
  and `train_stage2_cv.py` match the Linux workstation audit hashes.
- The original frozen 540-development/60-independent subject split and 10-fold
  development protocol (486 train/54 validation per fold) were verified on Linux,
  including the SHA-256 fingerprints referenced by the Stage3 code.
- **All 13 original Stage3 selected A2 core Python files match the A2
  final-development training contract exactly.** The only former GitHub
  discrepancy was three comment lines in `config_stage3.py`, now removed.
- A2 historical contract records 13 epochs, batch 8, LR `1.5e-4`,
  weight decay `2e-4`, seed 42, variance regularizer `0.0`, pair weight `0.2`.

## Outstanding verification / author decisions

1. Identify the **separate final-development Stage3 training runner** with
   SHA-256 `f658fe2f8712606108322c709bbf2f04e0e0b01011b06910664de77437435616`;
   this is **not** the same digest as the CV trainer.
2. Establish source/checkpoint/cache provenance for Stage2 full-development
   118-epoch fitting and Stage3 model inputs; resolve any source-specific
   `/255.0` versus raw-scale ambiguity without changing historical files.
3. Link the final A2 checkpoint, post-freeze inference and metric-generation
   scripts/records to the selected results; do not rerun the independent holdout
   merely to complete source bookkeeping.
4. Approve an explicit software license and authorship rights, and audit the
   entire Git history for credentials, personal identifiers, measurements, and
   accidentally tracked artifacts.
5. If obtainable, document the **historical** environment from run records;
   present Python/PyTorch/CUDA versions represent the **current workstation**.

Source authenticity and passing synthetic unit tests are not an independent
reproduction of manuscript numerical outcomes.
