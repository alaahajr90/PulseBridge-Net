# Test scope

Run synthetic unit checks with:

```bash
python -m unittest discover -s tests -v
```

Tests exercise Stage-1 and supplied v3 Stage-2 tensor contracts, selected
Stage-3 loss coefficients, original-code preservation and static protocol
expectations. They **do not** test Stage-2 v4 source (missing), original
HDF5 QC or actual holdout performance.

The audit that produced the accompanying ZIP ran on a CPU-only container;
actual Linux training still depends on restoring original files and paths.
