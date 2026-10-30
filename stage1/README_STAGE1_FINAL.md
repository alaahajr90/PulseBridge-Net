# PulseBridge-Net — Stage 1 FINAL Training Package

## Frozen scientific inputs

Stage 1 consumes **only** rows with `status=READY` from:

`Data_QC_v2/results/04_stage1_manifest.csv`

The package verifies the manifest SHA256 before training. Current frozen manifest:

`70dfba7b07d6a9cc679d07b183d2b7cff17ee6d59033b65bec44dd61b8163878`

This corresponds to:
- 3092 READY recordings
- 508 HOLD recordings
- 600/600 subjects represented
- 1200/1200 subject-state groups represented

Synchronization quality is **not** a Stage-1 inclusion gate.

## Subject protocol

On the first preflight, the code creates and freezes:

- `results/protocol/frozen_subject_split.json`
- `results/protocol/development_10folds.json`
- `results/protocol/protocol_manifest.json`
- `results/protocol/stage1_training_contract.json`

The split is generated from the original `db.csv` 600-subject roster:
- 60 subjects: locked blind test
- 540 subjects: development
- 10 CV folds inside development
- each fold: 486 train + 54 validation subjects

The split is NOT generated from QC-filtered recordings.

There is intentionally no command in this package that overwrites an existing
frozen protocol.

## Blind test

The blind test is **not evaluated in Stage 1**.

The final Stage-1 encoder is trained on the 540 development subjects only and
saved for Stage 2. The blind 60 subjects remain untouched for downstream
end-to-end evaluation.

## PPG policy

Stage-1 training dataset reads only `roi` and `clip_starts`.

It does **not** read PPG for:
- loss
- checkpoint selection
- early stopping
- diagnostics

Therefore Stage 1 remains strictly self-supervised with respect to physiology.

## Subject-balanced contrastive sampling

The READY recording counts differ among subjects. Therefore ordinary
`shuffle=True` would overweight subjects with more recordings.

The final sampler:
- allocates approximately equal samples per subject each epoch;
- subject exposure differs by at most one sample;
- randomly rotates through each subject's READY recordings;
- never places the same subject twice in one contrastive batch;
- generates stateless deterministic sample seeds.

This also reduces same-subject false negatives inside the contrastive batch.

## Validation

Every validation subject contributes exactly:
- one deterministic BEFORE recording;
- one deterministic AFTER recording.

Thus each subject and state contributes equally to validation loss. Camera choice
is deterministic and subject-specific.

## Runtime batch size

The old physical batch size 16 may not fit a 12-GB GPU for a `[3,128,64,64]`
3D-CNN clip. Preflight therefore performs a real forward+backward probe and
tries:

`16 -> 8 -> 4 -> 2`

The largest passing batch is frozen once in:

`results/preflight/resolved_runtime.json`

Training subsequently reuses that exact value. This is a hardware preflight,
not an epoch-by-epoch adaptive choice.

## Architecture contract

The encoder architecture is preserved. The final four Conv3d layers remain:

- `convblock2.0`
- `convblock2.3`
- `convblock3.0`
- `convblock3.3`

This preserves the Stage-2 fine-tuning contract.

## Checkpoint selection

Per fold:
1. primary: minimum validation SSL loss;
2. tie within `MIN_DELTA`: larger validation cosine gap.

No PPG and no blind-test metric participate in selection.

## Final model

After all 10 folds:
- collect 10 best epochs;
- use the median best epoch;
- initialize a fresh Stage-1 model;
- train on all 540 development subjects for exactly that many epochs;
- save:

`results/final_model/stage1_encoder_for_stage2.pth`

## Resume safety

`bash run_stage1.sh` uses `--resume`.

A checkpoint is resumed only if its stored training contract exactly matches:
- QC manifest hash
- frozen split hash
- frozen folds hash
- resolved physical batch size
- training hyperparameters
- key source-code hashes

Otherwise the run stops rather than mixing incompatible results.

## Run

```bash
cd /home/hajr/mcd_rppg/code_HR_RR_model/PulseBridge_Net/Stage_1_PulseRepresentation
bash run_stage1.sh
```
