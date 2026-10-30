# Stage 3 selected A2 (λvar=0)

Derived from the uploaded v20 source. The **only source change** relative to
`../stage3_v20_reference/` is in `config_stage3.py`:

```
V11_VARIANCE_WEIGHT = 0.05   # original v20 reference
V11_VARIANCE_WEIGHT = 0.0    # selected A2 manuscript configuration
```

All Stage-3 model, loss, training and paired residual code is preserved.
The Stage-2 waveform `W_STD=0.25` objective is not disabled by this Stage-3
change. This repository does NOT prove historical A2 runs employed precisely
this published file without the corresponding training contracts/checkpoints.

For actual runs: original scripts contain absolute historical paths and expect
preexisting frozen split, Stage-2 v4 caches and training manifests; do not run
on patient data until the path/provenance audit is complete.
