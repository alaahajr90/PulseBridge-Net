# Stage 2 — authentic Linux v4 source ↔ manuscript equations

Validated against 42-page manuscript `main(20261030-154034).pdf`,
Section 3.4, Eqs. (5–17). **This is the original v4 source, not the prior
v3-derived candidate.**

| Manuscript element | Original source implementation | Status |
|---|---|---|
| 128-frame ROI, unscaled pixels | `config_stage2.py`: `CLIP_LEN=128`, `ROI_SIZE=64`, `SCALE_TO_01=False` | verified config |
| 7 Conv3D transfer, BN frozen | `model_stage2.py`, `runtime_stage2.py` | architecture present |
| Dilated Conv1D decoder 1/2/4/8; 3 auxiliary heads | `model_stage2.py` | implemented |
| Waveform `[B,T]` | `Stage2RPPGModel.forward()['rppg']` | implemented |
| F7 latent `[B,T,64]` | `Stage2RPPGModel.forward_late_features(x)[-1]` | implemented, not `forward()['f7']` |
| Eqs. (5–10), waveform and physiological loss | `data_stage2.py`, `losses_stage2.py` | implemented |
| Eq. (11) per-sample population z-normalization | `losses_stage2._zshape` | implemented |
| Eq. (12) first difference, coefficient 0.15 | `normalized_d1_shape_per_sample`, `W_MORPH_D1` | implemented |
| Eq. (13) second temporal difference, coefficient 0.35 | `normalized_d2_shape_per_sample`, `W_MORPH_D2` | implemented |
| Eq. (14) multiscale increments, lags 1/2/4, coefficient 0.20 | `multiscale_shape_per_sample`, `W_MORPH_MULTISCALE` | implemented |
| Eqs. (15–17), weighted final + aux 0.15 | `RPPGPhysioCompositeLoss.forward` | implemented |
| AdamW, 120 max epochs, 18 patience | `config_stage2.py`, `train_stage2_cv.py` | implemented |
| 540/60 and 10 subject-exclusive folds | `protocol_stage2.py`; workstation SHA and counts | verified without exposing IDs |
| Stage2 final 118 epochs, batch16 | manuscript/training records, not this source ZIP | still requires authentic final checkpoint/run verification |
| Stage2 r and Stage3 BP MAEs | real-data checkpoints and outputs not supplied | not reproduced in this package |

**Important:** The v4 source is byte-accurate. Original absolute machine paths
require user setup outside CI. The independent 60-person test must never be
reused for feature/loss/model selection. See `STAGE2_V4_SOURCE_AUDIT.md`.
