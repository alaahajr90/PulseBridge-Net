# Data_QC_v2: audited source and recorded aggregate results

This report summarizes **the author's supplied archived QC outputs**;
it is not a new run of extraction or QC on the underlying videos.

- Source input: `Data_QC_v2.zip`, 43 ZIP entries, structurally intact.
- Source `extract_mcd_rppg_clips_compact.py`: SHA-256 `86162bc72f93819ae2892d12670ddf2a57481ca9acefef9566f7744c8848308f`;
  it **matches** the extractor hash in the supplied QC source contract.
- All 13 results files named in `08_qcv2_snapshot_hashes.json`
  matched their originally recorded SHA-256 values.
- 3,600/3,600 original video/recording slots have HDF5 results.
- 600 distinct subjects in the original roster and 1,200 subject–states.
- 44 decode failures were repaired in the archived run without modifying source videos.
- 0 structurally bad HDF5 recordings; 0 missing HDF5 recordings.
- ROI GOOD: 3,092; ROI HOLD: 508.
- Sync GOOD: 3,582; REVIEW: 9; BAD: 9.
- Stage 1 READY: 3,092 recordings (all 600 subjects represented).
- Stage 2 READY: 3,079 recordings (599 subjects; 6 subject–states absent).
- Stage 3 READY: 3,092 recordings (all 600 subjects represented).
- Canonical Stage 3 labels were finite in 3,600 HDF5 recordings.
- 0 inconsistent subject–state clinical label groups in 1,200 groups.

## Stage-specific inclusion policies

* Stage 1: structurally valid HDF5 and ROI QC GOOD; synchronized
  PPG quality is not an exclusion gate for self-supervised representation.
* Stage 2: structurally valid HDF5, ROI QC GOOD, and source `ppg_sync`
  QC GOOD for supervised rPPG waveform training/evaluation.
* Stage 3: structurally valid HDF5, ROI QC GOOD and finite SBP/DBP/stiffness
  labels; predicted rPPG is used and source synchronization is not a gate.

## Historical split verification remains open

The supplied `06_protocol_roster_audit.json` reports
`frozen_protocol_exists: false`. This shows that **that QC run did
not locate the two expected files** (`frozen_subject_split.json`,
`development_10folds.json`) at its configured path. It does **not**
demonstrate that those historical files never existed or that the
independent cohort was used improperly. The original split and
10-fold manifests must be recovered and checked before a full
reproducibility claim. Never regenerate a different fixed test split.

## Privacy and traceability

The original CSV reports and logs include subject identifiers,
sensitive physiological labels, and local file paths. They have been
intentionally **excluded** from this public-safe source bundle.
The original checksums are preserved in
[`provenance/QC_V2_SOURCE.json`](../provenance/QC_V2_SOURCE.json),
and the full original archive should be stored privately.

Full original preprocessing/QC has **not** been rerun in this package;
only source integrity and offline unit/static tests are checked.


## Read-only frozen-protocol check (on the original Linux workstation)

Run this only against the **historically existing** protocol files, without
regenerating them and without opening the independent evaluation labels:

```bash
python scripts/verify_frozen_protocol.py   --split /PATH/TO/ORIGINAL/frozen_subject_split.json   --folds /PATH/TO/ORIGINAL/development_10folds.json   --roster-csv /PATH/TO/ORIGINAL/db.csv
```

The tool prints only aggregate counts and SHA-256 fingerprints, never subject IDs.
Use the **already-existing frozen** JSON files; do not replace missing files
with a newly random-generated partition.
