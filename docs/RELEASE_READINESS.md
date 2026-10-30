# Public-release readiness — PulseBridge-Net (2026-10-30)

**Decision: NOT YET READY for an unconditional public code-availability claim.**
The staged repository is useful for internal review and can remain **Private**.
A carefully labelled *incomplete research preview* could technically be made
public after author/rights approval, but it must not be presented as a fully
reproducing implementation of the published results.

| Gate | Current evidence | Release state |
|---|---|---|
| Code present for Stage 1 | Provided source | Initial code audit only |
| Manuscript-aligned Stage 2 equations | Derived v3-to-v4 candidate; 10 synthetic tests previously passed | Historical v4 source/checkpoint provenance still missing |
| Selected Stage 3 A2 loss | `V11_VARIANCE_WEIGHT=0.0` in candidate | Dated A2 training/final-fit contracts not supplied |
| **Canonical preprocessing/Data_QC_v2** | README placeholder only | **BLOCKER** |
| Frozen 540/60 and 10-fold provenance | Some expected hashes stored in configs | **BLOCKER:** exact source files and validation missing |
| Stage-2 F7/rPPG cache traceability | Paths/hash claims in paper and configs | **BLOCKER:** verified artifacts not supplied |
| End-to-end subject-disjoint reproduction | Not executed from release bundle | **BLOCKER** for full-reproduction claim |
| License and collaborator rights | `LICENSE_DECISION.md` unresolved | **BLOCKER** for an open-source license claim |
| Data protection | Gitignore filters common file extensions | Manual entire-tree and Git-history review still required |
| Dependency versions | Suggested profile supplied | Historical environment lock not verified |
| GitHub Actions | Static test only; not a CUDA/data reproduction | Must inspect workflow result separately |

## Scientific integrity requirements

1. Retrieve genuine canonical `Data_QC_v2` source and explain HDF5 ROI/PPG
   content, clip alignment, QC thresholds, camera IDs and manifests. Do not
   substitute the legacy NPZ clip extractor without documenting the change.
2. Confirm the code/checkpoint provenance of Stage-2 historical **v4**.
   The candidate intentionally implements the manuscript loss but has not
   been shown to be the source of the published results.
3. Verify A2's frozen actual run contracts and the exact full-development
   training/inference setup. The variance-weight edit alone is not proof.
4. Validate matching subject-exclusive split and fold inheritance without
   opening or re-evaluating the independent cohort. Preserve the fixed 60
   subjects and their original one-time evaluation.
5. Audit all Git-tracked files *and Git history* for patient recordings,
   measurements, checkpoints, secrets and private paths. Confirm rights with
   all code contributors and choose a license deliberately.
6. Replace illustrative environment pins with historical version provenance
   only when backed by original logs/export. Distinguish model source tests
   from full evaluation in the README and article's Code Availability section.

## Final recommendation

Keep the repository **Private** while unresolved P0 gates remain. If published
as an *incomplete pre-release* with all limitations prominent, that is a
separate transparency decision and does not justify claiming the historical
figures have been independently reproduced.
