# PulseBridge-Net — supervisor source handoff (31 October 2026)

## Source provenance

- Original Linux `Data_QC_v2`/compact HDF5 extraction source integrated.
- Stage1 original author-supplied encoder source integrated.
- **Stage2 now uses genuine original v4 MorphologyPreserving source**, not the
  previously derived v3-to-v4 candidate. SHA-256 hashes for model/loss/config/
  trainer all match the Linux training-machine audit. A minor `inspect_stage2.py`
  manifest discrepancy is documented rather than concealed.
- Stage3 public-source candidate still derives from v20 with A2 variance loss
  disabled; the original Linux A2 final training contract confirms 13 epochs,
  LR 1.5e-4, weight decay 2e-4, variance=0.
- Frozen Stage1 split and folds match Stage3 expected cryptographic hashes:
  540 development, 60 independent, 10 folds x 486 train/54 val, no overlap.
- Source + model data-free GitHub Actions smoke tests have passed for the older
  candidate; the new authentic-v4 patch must pass a new GitHub run after commit.

## Remaining publication gates

1. Match historical Stage2 fold/final checkpoint SHA-256 and F7/rPPG caches,
   especially `/255` raw-pixel audit discrepancy.
2. Match Stage3 historical A2 training source/checkpoints and frozen end-to-end
   inference path to the selected-config experiment records.
3. Authorize license, review all Git-tracked files and Git history for data and
   credentials, and confirm rights clearance.
4. Record original historical training environment, if available, separately
   from the current workstation snapshot.

Until fully checked, describe this repository as *authenticated source code
under final provenance verification*, not an independently reproduced result.
There is no need to retrain or re-evaluate the independent holdout to close
source-level documentation gaps.
