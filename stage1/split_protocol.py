from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config_stage1 import (
    BLIND_TEST_FRACTION,
    DB_CSV,
    EXPECTED_TOTAL_SUBJECTS,
    FROZEN_FOLDS_JSON,
    FROZEN_SPLIT_JSON,
    K_FOLDS,
    PROTOCOL_MANIFEST_JSON,
    REQUIRE_600_SUBJECTS,
    SEED,
    STAGE1_QC_MANIFEST,
)
from io_utils import load_json, save_json, sha256_file


def _sid(x) -> str:
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except Exception:
        return s


def numeric_subject_sort(subjects):
    return sorted(subjects, key=lambda x: int(x) if str(x).isdigit() else str(x))


def load_original_subject_roster(db_csv: Path = DB_CSV) -> List[str]:
    subjects = set()
    with open(db_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            subjects.add(_sid(row["patient_id"]))
    roster = numeric_subject_sort(subjects)
    if REQUIRE_600_SUBJECTS and len(roster) != EXPECTED_TOTAL_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_SUBJECTS} original subjects, found {len(roster)}"
        )
    return roster


def roster_sha256(roster: List[str]) -> str:
    payload = "\n".join(map(str, roster)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_loaded_protocol(protocol: Dict, roster: List[str]) -> None:
    blind = set(map(str, protocol["blind_test_subjects"]))
    dev = set(map(str, protocol["development_subjects"]))
    original = set(map(str, roster))

    if blind & dev:
        raise RuntimeError("Frozen protocol corrupted: blind/development overlap.")
    if blind | dev != original:
        raise RuntimeError(
            "Frozen protocol no longer equals the original db.csv subject roster."
        )
    if protocol.get("seed") != SEED:
        raise RuntimeError("Frozen protocol seed differs from current scientific protocol.")
    if len(blind) != int(round(len(roster) * BLIND_TEST_FRACTION)):
        raise RuntimeError("Frozen blind-test subject count is inconsistent.")


def freeze_subject_protocol() -> Dict:
    FROZEN_SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    roster = load_original_subject_roster()

    if FROZEN_SPLIT_JSON.exists():
        protocol = load_json(FROZEN_SPLIT_JSON)
        _validate_loaded_protocol(protocol, roster)
        return protocol

    rng = np.random.default_rng(SEED)
    arr = np.asarray(roster, dtype=object)
    rng.shuffle(arr)

    n_test = int(round(len(roster) * BLIND_TEST_FRACTION))
    blind = numeric_subject_sort(map(str, arr[:n_test].tolist()))
    dev = numeric_subject_sort(map(str, arr[n_test:].tolist()))

    protocol = {
        "protocol_version": "PulseBridge_subject_level_v2_blind_locked",
        "seed": SEED,
        "source_roster": str(DB_CSV),
        "source_roster_sha256": roster_sha256(roster),
        "n_total_subjects": len(roster),
        "blind_test_fraction": BLIND_TEST_FRACTION,
        "n_blind_test_subjects": len(blind),
        "n_development_subjects": len(dev),
        "blind_test_subjects": blind,
        "development_subjects": dev,
        "blind_status": "LOCKED_NOT_EVALUATED_IN_STAGE1",
        "invariants": [
            "subject-level split only",
            "all recordings of one subject remain in one partition",
            "blind test selected from original db.csv roster before QC filtering",
            "blind test is absent from Stage-1 training and validation",
            "blind test is not evaluated in Stage 1",
            "QC never regenerates the subject partition",
        ],
    }
    save_json(FROZEN_SPLIT_JSON, protocol)
    return protocol


def _validate_loaded_folds(protocol: Dict, folds: Dict) -> None:
    blind = set(protocol["blind_test_subjects"])
    dev = set(protocol["development_subjects"])

    if int(folds["k_folds"]) != K_FOLDS:
        raise RuntimeError("Frozen fold count differs from current protocol.")

    validation_union = set()
    validation_counts = {}

    for f in folds["folds"]:
        tr = set(map(str, f["train_subjects"]))
        va = set(map(str, f["val_subjects"]))

        if tr & va or tr & blind or va & blind:
            raise RuntimeError(f"Subject leakage in frozen fold {f['fold']}.")
        if tr | va != dev:
            raise RuntimeError(f"Fold {f['fold']} does not cover development roster.")

        for s in va:
            validation_counts[s] = validation_counts.get(s, 0) + 1
        validation_union |= va

    if validation_union != dev:
        raise RuntimeError("Validation-fold union does not equal development roster.")
    if any(v != 1 for v in validation_counts.values()):
        raise RuntimeError("Each development subject must be validation exactly once.")


def freeze_development_folds(protocol: Dict) -> Dict:
    FROZEN_FOLDS_JSON.parent.mkdir(parents=True, exist_ok=True)

    if FROZEN_FOLDS_JSON.exists():
        folds = load_json(FROZEN_FOLDS_JSON)
        _validate_loaded_folds(protocol, folds)
        return folds

    dev = np.asarray(protocol["development_subjects"], dtype=object)
    rng = np.random.default_rng(SEED)
    rng.shuffle(dev)
    chunks = np.array_split(dev, K_FOLDS)

    all_dev = set(map(str, protocol["development_subjects"]))
    blind = set(map(str, protocol["blind_test_subjects"]))
    folds = []

    for i, chunk in enumerate(chunks, 1):
        val = set(map(str, chunk.tolist()))
        train = all_dev - val

        if train & val or train & blind or val & blind or train | val != all_dev:
            raise RuntimeError(f"Generated leakage in fold {i}.")

        folds.append(
            {
                "fold": i,
                "train_subjects": numeric_subject_sort(train),
                "val_subjects": numeric_subject_sort(val),
                "n_train_subjects": len(train),
                "n_val_subjects": len(val),
            }
        )

    payload = {
        "protocol_version": protocol["protocol_version"],
        "seed": SEED,
        "k_folds": K_FOLDS,
        "development_subjects_only": True,
        "folds": folds,
    }
    _validate_loaded_folds(protocol, payload)
    save_json(FROZEN_FOLDS_JSON, payload)
    return payload


def initialize_or_load_protocol() -> Tuple[Dict, Dict, Dict]:
    protocol = freeze_subject_protocol()
    folds = freeze_development_folds(protocol)
    roster = load_original_subject_roster()

    manifest = {
        "protocol_version": protocol["protocol_version"],
        "source_db_csv": str(DB_CSV),
        "source_db_sha256": sha256_file(DB_CSV),
        "source_roster_sha256": roster_sha256(roster),
        "stage1_qc_manifest": str(STAGE1_QC_MANIFEST),
        "stage1_qc_manifest_sha256": sha256_file(STAGE1_QC_MANIFEST),
        "frozen_split_json": str(FROZEN_SPLIT_JSON),
        "frozen_split_sha256": sha256_file(FROZEN_SPLIT_JSON),
        "frozen_folds_json": str(FROZEN_FOLDS_JSON),
        "frozen_folds_sha256": sha256_file(FROZEN_FOLDS_JSON),
        "protocol_summary": {
            "total_subjects": protocol["n_total_subjects"],
            "development_subjects": protocol["n_development_subjects"],
            "blind_test_subjects": protocol["n_blind_test_subjects"],
            "k_folds": K_FOLDS,
            "per_fold_ratio_of_original_roster": {
                "train": 0.81,
                "validation": 0.09,
                "blind_locked": 0.10,
            },
        },
        "blind_test_policy": "LOCKED_NOT_EVALUATED_IN_STAGE1",
    }
    save_json(PROTOCOL_MANIFEST_JSON, manifest)
    return protocol, folds, manifest


def assert_no_subject_leakage(protocol: Dict, folds: Dict) -> bool:
    _validate_loaded_protocol(protocol, load_original_subject_roster())
    _validate_loaded_folds(protocol, folds)
    return True
