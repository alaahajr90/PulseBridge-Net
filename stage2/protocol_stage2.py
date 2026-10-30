from __future__ import annotations

from typing import Dict, Tuple

from config_stage2 import (
    EXPECTED_FROZEN_FOLDS_SHA256,
    EXPECTED_FROZEN_SPLIT_SHA256,
    FROZEN_FOLDS_JSON,
    FROZEN_SPLIT_JSON,
    K_FOLDS,
    PROTOCOL_MANIFEST_JSON,
)
from io_utils import load_json, sha256_file


def load_stage1_protocol() -> Tuple[Dict, Dict, Dict]:
    for path in (
        FROZEN_SPLIT_JSON,
        FROZEN_FOLDS_JSON,
        PROTOCOL_MANIFEST_JSON,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required frozen Stage-1 protocol missing: {path}. "
                "Stage 2 must never generate a replacement split."
            )

    split_sha = sha256_file(FROZEN_SPLIT_JSON)
    folds_sha = sha256_file(FROZEN_FOLDS_JSON)

    if split_sha != EXPECTED_FROZEN_SPLIT_SHA256:
        raise RuntimeError(
            "Frozen Stage-1 split SHA256 changed.\n"
            f"Expected: {EXPECTED_FROZEN_SPLIT_SHA256}\n"
            f"Actual:   {split_sha}"
        )
    if folds_sha != EXPECTED_FROZEN_FOLDS_SHA256:
        raise RuntimeError(
            "Frozen Stage-1 folds SHA256 changed.\n"
            f"Expected: {EXPECTED_FROZEN_FOLDS_SHA256}\n"
            f"Actual:   {folds_sha}"
        )

    protocol = load_json(FROZEN_SPLIT_JSON)
    folds = load_json(FROZEN_FOLDS_JSON)
    manifest = load_json(PROTOCOL_MANIFEST_JSON)

    if int(folds["k_folds"]) != K_FOLDS:
        raise RuntimeError(f"Frozen K={folds['k_folds']}; expected K={K_FOLDS}.")

    blind = set(map(str, protocol["blind_test_subjects"]))
    dev = set(map(str, protocol["development_subjects"]))

    if blind & dev:
        raise RuntimeError("Frozen outer split contains blind/development overlap.")
    if len(blind) != 60 or len(dev) != 540 or len(blind | dev) != 600:
        raise RuntimeError("Frozen Stage-1 subject counts are not 60/540/600.")

    val_counts = {}
    for fold in folds["folds"]:
        tr = set(map(str, fold["train_subjects"]))
        va = set(map(str, fold["val_subjects"]))

        if tr & va or tr & blind or va & blind:
            raise RuntimeError(f"Subject leakage in frozen fold {fold['fold']}.")
        if tr | va != dev:
            raise RuntimeError(
                f"Fold {fold['fold']} does not exactly cover development subjects."
            )
        for s in va:
            val_counts[s] = val_counts.get(s, 0) + 1

    if set(val_counts) != dev or any(v != 1 for v in val_counts.values()):
        raise RuntimeError(
            "Each development subject must appear in validation exactly once."
        )

    return protocol, folds, manifest
