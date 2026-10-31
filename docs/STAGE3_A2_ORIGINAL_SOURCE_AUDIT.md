# Stage 3 A2 — Original Linux source authentication (2026-10-31)

**Finding:** all 13 Stage-3 Python files included in the final-development A2
training contract have **byte-identical SHA-256** to the files recovered from the
original Linux experiment directory `Ablation_Stage3_2026/A2_NoVarianceLoss/`.
The verified source archive has SHA-256
`db11ade32a9a98ea1ada5e9478e9ee24788f6a89b13ab3921665c7c180e460d1`.
See `../provenance/STAGE3_A2_ORIGINAL_SOURCE_SHA256.json` for all 13 hashes.

## Historical vs previously published candidate

Twelve files already matched the GitHub candidate exactly. Only
`stage3/config_stage3.py` differed: the GitHub candidate contained **three
additional comment lines**, explaining the variance loss ablation. Both
configurations had the **same Python abstract syntax tree** and values:
`V11_VARIANCE_WEIGHT=0.0`; no algorithm or runtime behavior was changed by
removing the comments. The current `stage3/config_stage3.py` now exactly matches
the Linux A2 source (`54361a9c33ee1094...`). No other Stage-3 scientific source
was modified in this patch.

## Historical final-development A2 contract

- Contract digest: `c0506f791ee523315f3f6905bf5fa6e210eebec6538f33028f89466d7e72b2a7`.
- 13 epochs; batch 8; seed 42; AdamW LR `1.5e-4`; weight decay `2e-4`.
- `V11_VARIANCE_WEIGHT=0.0`, paired loss coefficient `0.20`.
- Frozen split hash `d71d9c9e8ef75ef91f28cbac06b1b01de3325b95ecb3a992fb9ecb5afc7be739`;
  10-fold hash `628057141b979eb3af60595505a995b18d57e7dc4fd45e46d5d490ecb9f6c3e4`.
- Stage-2 full-development checkpoint digest recorded (not independently
  matched to checkpoint bytes): `42df926754a34c664ac6fcd1ff9e5281646309277bb203eab06187f1c96e87f2`.

**Remaining gap:** the final-development runner named by the contract's distinct
`trainer_sha256=f658fe2f8712606108322c709bbf2f04e0e0b01011b06910664de77437435616`
was **not located** in the searched Linux project. The hash of the generic
`train_stage3_cv.py` is different; do **not** relabel that fold-training program
as the final 13-epoch runner. The source-only ZIP also omits checkpoints,
run manifests, participant information and individual test results.

**Scope:** authentic source / training-contract audit, **not** a numerical
reproduction of published accuracy, and **not** a rerun of the fixed independent
holdout. Absolute paths in the historical configuration are intentionally
preserved to maintain source fingerprints; they are not portable defaults.
