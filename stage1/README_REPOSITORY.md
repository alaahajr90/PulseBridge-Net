# Stage 1: Self-supervised facial representation learning

Original source from uploaded Stage-1 final package, copied without algorithm changes.

- 7 Conv3D layers, 128-frame time dimension preserved, output `[B,128,64]`.
- Projection MLP is used only during Stage-1 SSL.
- Time-index positive pairs, one-direction margin-adjusted contrastive objective
  and temporal cosine regularizer, with temperature 0.07, margin 0.30,
  temporal weight 0.25.
- Augmentation preserves the original 0..255 RGB scale; no PPG labels in SSL.
- Uses **original frozen subject roster**, 540 development / 60 test and 10 CV folds.

Caution: supplied configuration uses original absolute filesystem paths and hashes.
Do not regenerate a new split over QC-filtered videos.
