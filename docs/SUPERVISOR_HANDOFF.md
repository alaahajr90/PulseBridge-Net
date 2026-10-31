# PulseBridge-Net: supervisor code handoff (31 October 2026)

## Current code-release state

This staging repository provides a **manuscript-aligned candidate** for the
PulseBridge-Net research pipeline. The code is suitable for **supervisor review**
with explicit scientific caveats; **it is not yet a verified reproduction** of
the published/development metrics. GitHub should remain **Private** until the
issues listed below are resolved, or be explicitly labelled a partial research
preview if the authors decide to disclose it before that point.

## Included and verified to source level

- Preprocessing: canonical compact HDF5 extractor from the archived source,
  identified by SHA-256, plus supplied Data_QC_v2 scripts. Aggregate QC
  evidence is documented; per-person measurement files are omitted.
- Stage 1: original author-supplied self-supervised Conv3D source; Python
  source tests do not establish training reproducibility.
- Stage 2: an **adapted candidate**, starting from the supplied v3 files,
  with morphology terms added to match the manuscript's equations; no claim
  of byte-for-byte identity with the original historical v4 trainer.
- Stage 3: v20-derived A2 candidate with variance coefficient set to zero;
  full training-contract/checkpoint identity not yet corroborated.
- Example environment specification, source audits, and offline tests.

## Evidence still needed to make a full code-availability claim

1. Verify original fixed 540-development / 60-independent subject assignments,
   and all ten inherited fold manifests, without regenerating the split.
2. Corroborate frozen Stage 2 **v4** checkpoint hashes and historical source
   or training contracts used to produce Stage 3's rPPG/F7 caches.
3. Corroborate the **A2** training/final-fitting runtime contracts and the
   subject-state aggregation/inference scripts behind the manuscript results.
4. Test the full pipeline on authorized non-blind development data, without
   repeating the one-time independent evaluation.
5. Approve rights, license, sensitive-data exclusion, full Git history, and
   example environment provenance before publishing.

## Manuscript wording while repository remains private

> The implementation and experimental configurations are maintained by the
> authors. A manuscript-aligned source-code candidate has been prepared for
> internal review; validation of complete reproducibility and code-release
> provenance is ongoing. The source repository is not yet publicly accessible.

Do not call the candidate a publicly available full reproduction until those
claims are supported by the actual released contents and evidence.
