# v20 — Frozen-Residual-D03

Controlled successor to v19.

- Blind remains locked.
- Absolute SBP/DBP/stiffness backbone and absolute loss are unchanged from the v18 lineage.
- D03 is `log_d1_rms`, summarized as six paired statistics.
- Final deployable correction remains antisymmetric: Before -= c/2, After += c/2.
- Critical change: SBP paired loss trains `c` against `true_delta - base_delta.detach()`.
- Absolute loss is computed on the uncorrected base SBP. Thus SBP paired/correction gradients cannot alter the absolute SBP backbone.
- DBP and rigidity paired branches remain v14-style.
- Checkpoint selection remains 80% absolute BP error + 20% FINAL corrected SBP delta error.

Run all 10 development folds with `python -u train_stage3_cv.py`.
