from __future__ import annotations
from config_stage3 import *
from io_utils import load_json, sha256_file

def load_frozen_protocol():
    for p in (FROZEN_SPLIT_JSON, FROZEN_FOLDS_JSON, PROTOCOL_MANIFEST_JSON):
        if not p.exists():
            raise FileNotFoundError(f"Missing Stage-1 frozen protocol: {p}")

    if sha256_file(FROZEN_SPLIT_JSON) != EXPECTED_FROZEN_SPLIT_SHA256:
        raise RuntimeError("Frozen Stage-1 subject split SHA256 mismatch.")
    if sha256_file(FROZEN_FOLDS_JSON) != EXPECTED_FROZEN_FOLDS_SHA256:
        raise RuntimeError("Frozen Stage-1 fold SHA256 mismatch.")

    split = load_json(FROZEN_SPLIT_JSON)
    folds = load_json(FROZEN_FOLDS_JSON)
    manifest = load_json(PROTOCOL_MANIFEST_JSON)

    blind = set(map(str, split["blind_test_subjects"]))
    dev = set(map(str, split["development_subjects"]))
    if len(blind) != 60 or len(dev) != 540 or blind & dev:
        raise RuntimeError("Frozen outer split failed 60/540 leakage audit.")
    if int(folds["k_folds"]) != K_FOLDS or len(folds["folds"]) != K_FOLDS:
        raise RuntimeError("Frozen K-fold structure mismatch.")

    seen_val = []
    for f in folds["folds"]:
        tr = set(map(str, f["train_subjects"]))
        va = set(map(str, f["val_subjects"]))
        if tr & va or tr & blind or va & blind:
            raise RuntimeError(f"Subject leakage in fold {f['fold']}")
        if tr | va != dev:
            raise RuntimeError(f"Fold {f['fold']} does not cover dev subjects exactly.")
        seen_val.extend(va)
    if len(seen_val) != 540 or len(set(seen_val)) != 540:
        raise RuntimeError("Every development subject must be validation exactly once.")
    return split, folds, manifest
