# Public-release readiness — PulseBridge-Net (31 Oct 2026)

**Scientific status:** authenticated core Stage-2 v4 and Stage-3 A2 source,
validated split hashes, and passing data-free unit/smoke tests; **not yet** a
fully independently reproduced end-to-end pipeline.

| Gate | Verified evidence | Status |
|---|---|---|
| Preprocessing | original HDF5 extractor + Data_QC_v2 source | source verified |
| Stage 1 | original self-supervised source | source provided |
| Stage 2 | genuine Linux v4 model/loss/config/CV trainer SHA-256 | source verified |
| Stage 3 A2 | 13/13 original Linux Python source hashes match final A2 contract | **source verified** |
| Original split | 540 development/60 independent; 10 folds 486/54, no overlap; SHA-256 matches | verified |
| Final A2 training config | 13 epochs, batch8, LR 1.5e-4, WD 2e-4, variance 0, pair 0.2 | verified from contract |
| Stage2 final checkpoint & F7/rPPG caches | cross-stage source, weight & cache lineage | pending |
| Separate A2 13-epoch final trainer | contract records SHA-256 but historical script not located | pending |
| Final checkpoint, inference and metrics | full chain not independently reproduced | pending |
| Testing | CPU synthetic/smoke unit tests only | passed locally; confirm latest GitHub Actions after push |
| Privacy and open-source rights | verify all Git history, licensed dependencies, contributor consent | pending |

**No claim of full independent reproduction** is supported by SHA-256 source
matches or synthetic tests alone. Original absolute Linux paths are retained
to preserve historical file fingerprints. Original Stage2 archival
`inspect_stage2.py` manifest checksum discrepancy remains documented.

**Decision:** keep the existing private repository for supervisor review until
rights/privacy gates and the outstanding artifact lineage are reconciled.
Alternatively, public release as an explicitly *partial source implementation*
requires careful disclosure of limitations and rights/secret scanning.
No re-evaluation of the independent holdout is necessary for these audits.
