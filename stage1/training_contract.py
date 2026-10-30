from __future__ import annotations

import json
from pathlib import Path

from config_stage1 import *
from io_utils import save_json, sha256_file


CONTRACT_JSON = OUTPUT_ROOT / "protocol" / "stage1_training_contract.json"


def build_training_contract(protocol_manifest, runtime):
    code_files = [
        STAGE1_ROOT / "config_stage1.py",
        STAGE1_ROOT / "data_stage1.py",
        STAGE1_ROOT / "model_stage1.py",
        STAGE1_ROOT / "split_protocol.py",
        STAGE1_ROOT / "train_stage1_cv.py",
    ]

    code_hashes = {}
    for p in code_files:
        if not p.exists():
            raise FileNotFoundError(p)
        code_hashes[p.name] = sha256_file(p)

    payload = {
        "contract_version": "PulseBridge_Stage1_FINAL_v3",
        "seed": SEED,
        "stage1_qc_manifest": str(STAGE1_QC_MANIFEST),
        "stage1_qc_manifest_sha256": sha256_file(STAGE1_QC_MANIFEST),
        "split_sha256": protocol_manifest["frozen_split_sha256"],
        "folds_sha256": protocol_manifest["frozen_folds_sha256"],
        "batch_size": int(runtime["batch_size"]),
        "epochs_max": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "temperature": TEMPERATURE,
        "margin": MARGIN,
        "temporal_weight": TEMPORAL_WEIGHT,
        "sampling": {
            "training": "subject-balanced; unique subject per contrastive batch",
            "validation": "one deterministic before + one deterministic after recording per subject",
        },
        "blind_policy": "LOCKED_NOT_EVALUATED_IN_STAGE1",
        "ppg_used_in_stage1_training": False,
        "code_sha256": code_hashes,
    }
    save_json(CONTRACT_JSON, payload)
    return payload
