# Required original records before claiming full publication reproducibility

**Priority P0** — exact original `Stage_2_rPPG_FineTuning_v4_MorphologyPreserving/`
source, training configuration, data reader, final loss with normalized first/second
and multiscale derivatives, and its frozen CV/final checkpoint fingerprints.
The supplied file named `code stage2_new.zip` is **v3**, not v4.

**Priority P0** — `Data_QC_v2` scripts used to create exact HDF5 `roi`, `ppg`,
`clip_starts`, and the three `04_stage{1,2,3}_manifest.csv` files; provide
manifest hashes and decision logic including frame-gap handling. Current
extractors write per-clip `.npz`, which is **not** the training input format.

**Priority P0** — frozen `frozen_subject_split.json`,
`development_10folds.json`, and `protocol_manifest.json` (or a privacy-safe
reproduction protocol with verified hashes). Do not silently regenerate
an alternative 60-subject holdout.

**Priority P0** — the actual A2/selected Stage-3 config/contract from each of the
10 folds (variance weight 0), as well as original full-development Stage-2
118-epoch / Stage-3 13-epoch scripts and the independent inference pipeline.

**Priority P1** — reproducible environment lock, checkpoint hash manifests,
fold metrics, blinded prediction artifacts and cached Stage2 F7 fingerprints
when redistribution and privacy policy permit.

**Priority P1** — license and rights clearance before releasing publicly.

All missing materials should be retrieved from completed experiments **without
rerunning the independent test or changing the paper's evaluation protocol**.


Stage2 now has a derived code implementation that matches the paper loss definitions; original v4 historical source and checkpoint identity still need checking.
