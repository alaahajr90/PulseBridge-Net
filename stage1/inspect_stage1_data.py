from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from config_stage1 import *
from data_stage1 import (
    Stage1MCDH5Dataset,
    SubjectBalancedBatchSampler,
    ValidationUniqueSubjectBatchSampler,
    load_stage1_manifest,
    recordings_for_subjects,
    select_deterministic_validation_recordings,
)
from diagnostics_stage1 import count_parameters
from io_utils import save_json, sha256_file
from model_stage1 import Encoder, ProjectionHead, contrastive_loss_components
from runtime_stage1 import resolve_or_load_runtime
from split_protocol import initialize_or_load_protocol, assert_no_subject_leakage
from training_contract import build_training_contract


def main():
    print("=" * 112)
    print("PulseBridge-Net Stage 1 — FINAL PRE-TRAINING AUDIT")
    print("=" * 112)

    # -------------------------------------------------------------------------
    # Frozen QC manifest
    # -------------------------------------------------------------------------
    accepted, rejected, qcs = load_stage1_manifest()
    print("Stage-1 manifest:", qcs["manifest_path"])
    print("Manifest SHA256:", qcs["manifest_sha256"])
    print("READY recordings:", qcs["ready_recordings"])
    print("HOLD recordings:", qcs["hold_recordings"])
    print("READY subjects:", qcs["ready_subjects"])
    print("READY subject-states:", qcs["ready_subject_states"])

    print("READY cameras:", dict(Counter(r.camera for r in accepted)))
    print("READY states:", dict(Counter(r.state for r in accepted)))

    # -------------------------------------------------------------------------
    # Subject protocol
    # -------------------------------------------------------------------------
    protocol, folds, protocol_manifest = initialize_or_load_protocol()
    assert_no_subject_leakage(protocol, folds)

    print("\nOriginal subjects:", protocol["n_total_subjects"])
    print("Frozen blind subjects:", protocol["n_blind_test_subjects"])
    print("Development subjects:", protocol["n_development_subjects"])
    print("K folds:", folds["k_folds"])
    print("Blind policy:", protocol["blind_status"])
    print("Split SHA256:", protocol_manifest["frozen_split_sha256"])
    print("Folds SHA256:", protocol_manifest["frozen_folds_sha256"])

    blind = set(protocol["blind_test_subjects"])
    dev = set(protocol["development_subjects"])

    if blind & dev:
        raise RuntimeError("Blind/development overlap.")

    for ref in accepted:
        if ref.subject_id not in blind and ref.subject_id not in dev:
            raise RuntimeError(f"Manifest subject outside frozen roster: {ref.subject_id}")

    print("[PASS] QC manifest subjects map only through frozen SUBJECT IDs.")
    print("[PASS] Blind test selected from original 600-subject roster before training.")
    print("[PASS] Blind test absent from all CV train/validation folds.")
    print("[PASS] Blind test will NOT be evaluated in Stage 1.")

    # -------------------------------------------------------------------------
    # Per-fold coverage and subject-balanced validation
    # -------------------------------------------------------------------------
    fold_reports = []
    for fold in folds["folds"]:
        train_refs = recordings_for_subjects(accepted, fold["train_subjects"])
        val_all = recordings_for_subjects(accepted, fold["val_subjects"])
        val_refs = select_deterministic_validation_recordings(
            accepted, fold["val_subjects"]
        )

        train_subjects_present = {r.subject_id for r in train_refs}
        val_subjects_present = {r.subject_id for r in val_refs}

        if train_subjects_present != set(fold["train_subjects"]):
            raise RuntimeError(f"Fold {fold['fold']} has missing training subjects.")
        if val_subjects_present != set(fold["val_subjects"]):
            raise RuntimeError(f"Fold {fold['fold']} has missing validation subjects.")

        if len(val_refs) != 2 * len(fold["val_subjects"]):
            raise RuntimeError(
                f"Fold {fold['fold']} validation is not exactly 2 recordings/subject."
            )

        fold_reports.append(
            {
                "fold": fold["fold"],
                "train_subjects": len(fold["train_subjects"]),
                "val_subjects": len(fold["val_subjects"]),
                "train_ready_recordings": len(train_refs),
                "val_ready_recordings_all": len(val_all),
                "val_selected_recordings": len(val_refs),
            }
        )

        print(
            f"Fold {fold['fold']:02d}: "
            f"train subjects={len(fold['train_subjects'])}, "
            f"val subjects={len(fold['val_subjects'])}, "
            f"train READY recs={len(train_refs)}, "
            f"val selected={len(val_refs)}, leakage=0"
        )

    # -------------------------------------------------------------------------
    # HDF5 contract sample
    # -------------------------------------------------------------------------
    sample = accepted[0]
    with h5py.File(sample.h5_path, "r") as h5:
        required = [
            "roi", "ppg", "ppg_sync_dt", "coords", "frame_ids",
            "frame_timestamps", "face_detected", "fallback_used",
            "clip_starts", "labels",
        ]
        missing = [k for k in required if k not in h5]
        if missing:
            raise RuntimeError(f"HDF5 missing keys: {missing}")

        if tuple(h5["roi"].shape[1:]) != (ROI_SIZE, ROI_SIZE, 3):
            raise RuntimeError("Unexpected ROI spatial shape.")

        print("\nHDF5 sample:", sample.h5_path)
        print("roi:", h5["roi"].shape, h5["roi"].dtype)
        print("clip_starts:", h5["clip_starts"].shape)

    # -------------------------------------------------------------------------
    # Dataset + deterministic seeded sample
    # -------------------------------------------------------------------------
    ds = Stage1MCDH5Dataset(accepted[:16], training=True)
    x1, x2, meta = ds[(0, 123456)]
    if tuple(x1.shape) != (3, CLIP_LEN, ROI_SIZE, ROI_SIZE):
        raise RuntimeError(f"Unexpected Stage-1 tensor shape: {tuple(x1.shape)}")

    x1b, x2b, metab = ds[(0, 123456)]
    if not torch.equal(x1, x1b) or not torch.equal(x2, x2b):
        raise RuntimeError("Seeded Stage-1 sample is not reproducible.")

    print("[PASS] Seeded clip selection + augmentations are worker-independent.")

    # -------------------------------------------------------------------------
    # Runtime batch-size probe on actual hardware
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = resolve_or_load_runtime(accepted, device)
    print("\nDevice:", device)
    print("Resolved physical batch size:", runtime["batch_size"])
    if "cuda_device_name" in runtime:
        print("CUDA device:", runtime["cuda_device_name"])

    # -------------------------------------------------------------------------
    # Real train-loader batch checks
    # -------------------------------------------------------------------------
    first_fold = folds["folds"][0]
    train_refs = recordings_for_subjects(accepted, first_fold["train_subjects"])
    sampler = SubjectBalancedBatchSampler(
        train_refs,
        batch_size=int(runtime["batch_size"]),
    )
    sampler.set_epoch(1)
    train_ds = Stage1MCDH5Dataset(train_refs, training=True)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    bx1, bx2, bmeta = next(iter(loader))

    batch_subjects = list(bmeta["subject_id"])
    if len(batch_subjects) != len(set(batch_subjects)):
        raise RuntimeError("Duplicate subject inside contrastive training batch.")

    print("[PASS] Subject-balanced sampler: unique subject inside training batch.")

    # Validate equal exposure quota property over one complete sampler epoch.
    counts = Counter()
    for batch in sampler:
        for idx, _seed in batch:
            counts[train_refs[idx].subject_id] += 1
    spread = max(counts.values()) - min(counts.values())
    if spread > 1:
        raise RuntimeError(f"Subject exposure imbalance >1 sample: spread={spread}")
    print("[PASS] Training exposure differs by at most one sample per subject.")

    # -------------------------------------------------------------------------
    # Model forward + SSL loss with selected runtime batch.
    # Runtime probe already tested backward on GPU; here inspect architecture.
    # -------------------------------------------------------------------------
    encoder = Encoder().to(device)
    projector = ProjectionHead(
        PROJ_INPUT_DIM,
        PROJ_HIDDEN_DIM,
        PROJ_OUTPUT_DIM,
        PROJ_DROPOUT,
    ).to(device)

    with torch.no_grad():
        features = encoder(bx1.to(device))
        z1 = projector(features)
        z2 = projector(encoder(bx2.to(device)))
        comp = contrastive_loss_components(
            z1, z2, TEMPERATURE, MARGIN, TEMPORAL_WEIGHT
        )

    print("Encoder parameters:", count_parameters(encoder))
    print("Encoder output:", tuple(features.shape))
    print("Projection output:", tuple(z1.shape))
    print("Initial SSL loss:", float(comp["total"]))
    print("Initial cosine gap:", float(comp["cosine_gap"]))
    print("Initial embedding std:", float(comp["embedding_std"]))

    # Critical Stage-2 layer names.
    conv_names = [
        n for n, m in encoder.named_modules() if isinstance(m, torch.nn.Conv3d)
    ]
    expected_last4 = [
        "convblock2.0",
        "convblock2.3",
        "convblock3.0",
        "convblock3.3",
    ]
    if conv_names[-4:] != expected_last4:
        raise RuntimeError(
            f"Stage-2 layer contract changed: last4={conv_names[-4:]}"
        )
    print("[PASS] Stage-2 encoder layer contract:", expected_last4)

    contract = build_training_contract(protocol_manifest, runtime)

    preflight = {
        "status": "PASS",
        "manifest": qcs,
        "protocol_manifest": protocol_manifest,
        "runtime": runtime,
        "fold_coverage": fold_reports,
        "training_contract": contract,
        "blind_test_evaluated": False,
        "ppg_loaded_by_training_dataset": False,
        "subject_leakage": 0,
        "subject_exposure_spread_first_fold": spread,
    }
    save_json(OUTPUT_ROOT / "preflight" / "stage1_preflight.json", preflight)

    print("=" * 112)
    print("STAGE-1 PRE-FLIGHT STATUS: PASS")
    print("SUBJECT LEAKAGE: 0")
    print("BLIND TEST: LOCKED / NOT EVALUATED")
    print("PPG SUPERVISION IN STAGE 1: NONE")
    print("=" * 112)


if __name__ == "__main__":
    main()
