# Stage 2 v4 — original source provenance (31 Oct 2026)

The source ZIP uploaded from the Linux training workstation contains 17 entries,
including 13 Python files. Four principal source SHA-256 digests matched the
previous training-machine audit **exactly**:

| File | SHA-256 |
|---|---|
| `model_stage2.py` | `a86f2a144785ee44f6d81161e5f041109d7c1430860603592671a9d5e33bdd94` |
| `losses_stage2.py` | `74606bbbb98f30c9c0cf4ec8cde73a3699269962d7825ecbb6a3472fbd9cb27f` |
| `config_stage2.py` | `f74ee2f733cf056cf8bbf1754cc2eaaf003881aa968d303043f2dfb05bbd8160` |
| `train_stage2_cv.py` | `8dfbc6ac0eb2584d758538251445b06e68e570ab46dab85a447076eabfa8360a` |

**Integrity exception**: `PACKAGE_MANIFEST.json` inside the provided archive
records a different digest for `inspect_stage2.py` (`77433c...`) from the actual
archived file (`8ba6ad...`). This discrepancy is transparently documented;
the manifest must not be treated as an unquestioned source of truth. Actual
archived file hashes are recorded in `provenance/STAGE2_V4_ORIGINAL_SOURCE_SHA256.json`.
No scientific source file was edited during incorporation.

**Historical documents:** The original README and shell runner were authored
at an earlier *development-only CV* phase and mention a prior blind-test status
and a no-final-fit rule. Their statements do not supersede the manuscript's
later one-time independent evaluation and completed final fits. A corrected
README documents this distinction; the CV runner is preserved as a historical
script, not offered as end-to-end model reproduction.

**Interface:** The authentic Stage2 `forward()` returns `rppg` and auxiliary
waveforms, while `forward_late_features(x)[-1]` exposes `F7`. The earlier
derived candidate exposed an additional `forward()['f7']` key. CI tests have
been updated for the genuine interface, without altering the authentic model.

**Remaining gap:** the original full-dev Stage2 v4 checkpoint fingerprints
and cross-stage rPPG/F7 cache hashes must be correlated with the archived
Stage3 runtime contracts and manuscript. No new model training, split generation
or blind-test inference was performed as part of this source audit.
