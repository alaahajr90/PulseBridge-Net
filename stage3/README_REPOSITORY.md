# Stage 3 — authenticated original selected A2 source

The 13 core Python source files in this directory are the **exact originals**
from the Linux `A2_NoVarianceLoss` experiment, with byte-identical SHA-256
hashes to the final-development A2 training contract. See
[`docs/STAGE3_A2_ORIGINAL_SOURCE_AUDIT.md`](../docs/STAGE3_A2_ORIGINAL_SOURCE_AUDIT.md)
and [`provenance/STAGE3_A2_ORIGINAL_SOURCE_SHA256.json`](../provenance/STAGE3_A2_ORIGINAL_SOURCE_SHA256.json).

**A2 selected objective**: `V11_VARIANCE_WEIGHT=0.0` (where the reference v20
configuration used `0.05`). The Stage-2 waveform standard-deviation objective
remains unchanged. Historical final-development A2 contract: 13 epochs,
physical batch size 8, learning rate `1.5e-4`, weight decay `2e-4`, seed 42,
paired-loss coefficient 0.20. The `train_stage3_cv.py` file implements the
cross-validation stage; the separately fingerprinted final-development trainer
was not located in the searched Linux project.

Historical filenames with older version suffixes remain for internal imports;
these should not be interpreted as multiple simultaneously selected models.
Source paths point to the original Linux workstation and require deliberate
mapping to run on another system. **No private subject data, weights, caches,
or frozen test outcomes are redistributed.** Source-level authenticity alone
does not establish full numerical reproducibility of reported metrics.
