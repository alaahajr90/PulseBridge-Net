# Stage 2: code ↔ manuscript crosswalk

**Basis:** 42-page `main(20261029-164645).pdf` manuscript, Sections 3.2 and 3.4,
Eqs. (5)–(17), Table 1, Section 4.2. Checked against the received
`code stage2_new.zip` FINAL_v3 source. The new code is a **derived candidate**.

| Manuscript | Candidate source | Status |
|---|---|---|
| ROI `[B,3,128,64,64]`, RGB intensity unscaled | `data_stage2.py`, `SCALE_TO_01=False` | code aligned; preprocessing not independently verified |
| Seven Conv3D encoder transferred from Stage 1 | `model_stage2.py` | aligned |
| All Conv3D trainable, first 3 LR 1e-5, last 4 LR 5e-5, frozen BN, decoder LR 2e-4 | `model_stage2.py`, `runtime_stage2.py`, `config_stage2.py` | aligned |
| Attention-free temporal TCN, dilation 1/2/4/8, 3 aux heads | `model_stage2.py` | aligned |
| `[B,128]` reconstructed waveform and `[B,128,64]` F7 | `model_stage2.py` `forward()` | both keys exposed; cache generation still missing |
| Eq. (5), per-segment standardized PPG | `data_stage2.py` | aligned |
| Eq. (6) Pearson r, Eq. (7) SmoothL1 first derivative | `losses_stage2.py` | aligned |
| Eqs. (8)–(9), NFFT512 band 0.5–4 Hz spectral distance/out-of-band penalty | `losses_stage2.py` | aligned |
| Eq. (10), target/estimate population-std mismatch | `losses_stage2.py` | aligned |
| Eq. (11), Z(v), epsilon=1e-6, population std | `z_normalize` | newly implemented |
| Eq. (12), normalized first differences `Lm1` | `morphology_terms_per_sample` | newly implemented weight .15 |
| Eq. (13), normalized *second temporal differences* `Lm2` | `second_difference` | newly implemented weight .35 |
| Eq. (14), increments with lags 1/2/4 `Lms` | `morphology_terms_per_sample` | newly implemented weight .20 |
| Eqs. (15)–(17), final + 0.15 mean auxiliary losses | `RPPGPhysioCompositeLoss.forward()` | implemented, synthetic equality tests |
| Max120 epochs, patience18, scheduler factor.5/patience6 | config/runtime/trainer | code aligned |
| Fixed split 60 blind / 540 dev and inherited 10 folds | `protocol_stage2.py` + hash checks | original frozen JSON not present in delivery |
| Full development Stage2 final run 118 epochs, batch16 (recorded) | training final-fit uses median CV best epoch and runtime batch | **unverified**; no original contracts/checkpoints to prove historical run |
| Stage2 input scale in third-party v4 OOF cache audit | earlier archive's cache builder had pixel /255 path | **open discrepancy; raw oldscale pipeline must be checked** |
| Stage2 CV waveform r=0.6507±0.0268 and downstream BP MAEs | no original predictions/checkpoints attached | **NOT REPRODUCED** |

## Source history and scientific limits

1. Original delivered v3 files have independent SHA256 fingerprints (see
   `provenance/STAGE2_V3_SOURCE_SHA256.json`).
2. No historical v4 checkpoints were used or modified; no holdout accessed.
3. This candidate implements manuscript mathematical definitions to the extent
   that they are specified. It does **not** establish that identical code,
   data pipeline, seeds, checkpoint selection, and recorded results were used.
4. Stage3's original `config_stage3.py` still points to the historical Stage2
   path and historical caches; it is **not connected** to this new candidate by
   merely adding Stage2 files into the GitHub repository.
5. To call this a verified reproduction, compare historical v4 file hashes or
   run the frozen development experiments and verify their outputs without
   refitting based on the independent test cohort.
