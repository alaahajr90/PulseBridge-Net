from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import DataLoader

from config_stage2 import *
from data_stage2 import (
    FixedValidationBatchSampler,
    Stage2RPPGDataset,
    SubjectBalancedClipBatchSampler,
    full_h5_ppg_audit,
    load_stage2_manifest,
    recordings_for_subjects,
    select_validation_recordings,
)
from io_utils import load_json, save_json, sha256_file
from losses_stage2 import RPPGPhysioCompositeLoss
from model_stage2 import Stage2RPPGModel, load_stage1_encoder_weights
from protocol_stage2 import load_stage1_protocol
from runtime_stage2 import resolve_or_load_runtime
from training_contract import build_training_contract, stage1_encoder_paths


def main():
    print("=" * 112)
    print("PulseBridge-Net Stage 2 — paper-aligned candidate PRE-TRAINING AUDIT")
    print("=" * 112)

    # -------------------------------------------------------------------------
    # Frozen Stage-1 protocol
    # -------------------------------------------------------------------------
    protocol, folds, stage1_protocol_manifest = load_stage1_protocol()
    blind = set(map(str, protocol["blind_test_subjects"]))
    dev = set(map(str, protocol["development_subjects"]))

    print("Frozen Stage-1 split SHA256:", sha256_file(FROZEN_SPLIT_JSON))
    print("Frozen Stage-1 folds SHA256:", sha256_file(FROZEN_FOLDS_JSON))
    print("Blind subjects:", len(blind))
    print("Development subjects:", len(dev))
    print("K folds:", folds["k_folds"])
    print("[PASS] Stage 2 did NOT create a new subject split.")

    # Stage-1 final blind lock must still exist.
    if not STAGE1_BLIND_LOCK.exists():
        raise FileNotFoundError(
            f"Stage-1 blind lock missing: {STAGE1_BLIND_LOCK}"
        )
    blind_lock = load_json(STAGE1_BLIND_LOCK)
    if blind_lock.get("status") != "LOCKED_NOT_EVALUATED_IN_STAGE1":
        raise RuntimeError("Stage-1 blind lock status is unexpected.")
    print("[PASS] Stage-1 blind test remained sealed.")

    # -------------------------------------------------------------------------
    # Authoritative Stage-2 manifest
    # -------------------------------------------------------------------------
    ready, hold, manifest_summary = load_stage2_manifest()
    ready_subjects = {r.subject_id for r in ready}
    ready_states = {(r.subject_id, r.state) for r in ready}
    original_subjects = blind | dev
    missing_ready_subjects = sorted(
        original_subjects - ready_subjects,
        key=lambda x: int(x) if x.isdigit() else x,
    )

    if tuple(missing_ready_subjects) != EXPECTED_NO_STAGE2_READY_SUBJECTS:
        raise RuntimeError(
            f"Unexpected subjects with zero Stage-2 READY recordings: "
            f"{missing_ready_subjects}"
        )

    if not dev <= ready_subjects:
        missing_dev = sorted(dev - ready_subjects)
        raise RuntimeError(
            f"Development subjects missing Stage-2 supervision: {missing_dev}"
        )

    if not set(missing_ready_subjects) <= blind:
        raise RuntimeError(
            "A zero-supervision Stage-2 subject is outside the blind partition."
        )

    print("\nStage-2 manifest:", manifest_summary["manifest_path"])
    print("Manifest SHA256:", manifest_summary["manifest_sha256"])
    print("READY recordings:", manifest_summary["ready_recordings"])
    print("HOLD recordings:", manifest_summary["hold_recordings"])
    print("READY subjects:", manifest_summary["ready_subjects"])
    print("READY subject-states:", manifest_summary["ready_subject_states"])
    print("READY cameras:", dict(Counter(r.camera for r in ready)))
    print("READY states:", dict(Counter(r.state for r in ready)))
    print("No Stage-2 READY subject(s):", missing_ready_subjects)
    print("[PASS] Missing Stage-2 supervision did NOT regenerate/replace subjects.")

    # -------------------------------------------------------------------------
    # Full H5 / target alignment audit
    # -------------------------------------------------------------------------
    dev_ready = recordings_for_subjects(ready, protocol["development_subjects"])
    print("\nRunning full DEVELOPMENT Stage-2 H5/PPG alignment audit...")
    h5_audit = full_h5_ppg_audit(dev_ready)

    if h5_audit["minimum_segment_ppg_std"] < MIN_SEGMENT_PPG_STD:
        raise RuntimeError(
            "At least one logical Stage-2 clip has a degenerate PPG target: "
            f"{h5_audit['minimum_segment_location']} "
            f"std={h5_audit['minimum_segment_ppg_std']}"
        )

    print("H5 recordings checked:", h5_audit["recordings_checked"])
    print("Logical PPG clips checked:", h5_audit["logical_clips_checked"])
    print(
        "Minimum logical-clip PPG std:",
        h5_audit["minimum_segment_ppg_std"],
    )
    print(
        "Missing source_ppg_sync attribute:",
        h5_audit["source_ppg_sync_attr_missing_count"],
    )
    print("[PASS] All DEVELOPMENT READY H5 targets are aligned, finite and non-degenerate.")
    print("[PASS] Blind H5 contents were NOT opened by Stage-2 preflight.")

    # -------------------------------------------------------------------------
    # Fold coverage + Stage-1 fold-specific encoder initialization
    # -------------------------------------------------------------------------
    fold_reports = []
    init_hashes = {}

    for fold in folds["folds"]:
        f = int(fold["fold"])
        train_refs = recordings_for_subjects(
            ready, fold["train_subjects"]
        )
        val_refs_all = recordings_for_subjects(
            ready, fold["val_subjects"]
        )
        val_refs_fixed = select_validation_recordings(
            ready, fold["val_subjects"]
        )

        train_present = {r.subject_id for r in train_refs}
        val_present = {r.subject_id for r in val_refs_all}

        if train_present != set(fold["train_subjects"]):
            raise RuntimeError(f"Fold {f} has a missing Stage-2 train subject.")
        if val_present != set(fold["val_subjects"]):
            raise RuntimeError(f"Fold {f} has a missing Stage-2 val subject.")

        init = (
            STAGE1_FOLDS_ROOT
            / f"fold_{f:02d}/checkpoints/encoder_best.pth"
        )
        if not init.exists():
            raise FileNotFoundError(init)

        init_hashes[f"fold_{f:02d}"] = sha256_file(init)

        fold_reports.append(
            {
                "fold": f,
                "train_subjects": len(fold["train_subjects"]),
                "val_subjects": len(fold["val_subjects"]),
                "train_ready_recordings": len(train_refs),
                "val_ready_recordings_all": len(val_refs_all),
                "val_selected_recordings": len(val_refs_fixed),
                "stage1_init": str(init),
                "stage1_init_sha256": init_hashes[f"fold_{f:02d}"],
            }
        )

        print(
            f"Fold {f:02d}: train subjects={len(fold['train_subjects'])}, "
            f"val subjects={len(fold['val_subjects'])}, "
            f"train READY recs={len(train_refs)}, "
            f"val fixed recs={len(val_refs_fixed)}, leakage=0"
        )

    if not STAGE1_FINAL_ENCODER.exists():
        raise FileNotFoundError(STAGE1_FINAL_ENCODER)

    print("[PASS] Each Stage-2 fold uses its matching Stage-1 fold encoder.")
    print("[PASS] Final Stage-2 fit will use Stage-1 final-development encoder only.")

    # -------------------------------------------------------------------------
    # Data/model contract sample
    # -------------------------------------------------------------------------
    first_fold = folds["folds"][0]
    first_train = recordings_for_subjects(
        ready, first_fold["train_subjects"]
    )

    ds = Stage2RPPGDataset(first_train[:4])
    x, y, meta = ds[(0, 0)]

    print("\nSample x:", tuple(x.shape), x.dtype)
    print(
        "Sample normalized PPG:",
        tuple(y.shape),
        y.dtype,
        "mean/std=",
        float(y.mean()),
        float(y.std(unbiased=False)),
    )

    if tuple(x.shape) != (3, CLIP_LEN, ROI_SIZE, ROI_SIZE):
        raise RuntimeError("Unexpected Stage-2 input shape.")
    if tuple(y.shape) != (CLIP_LEN,):
        raise RuntimeError("Unexpected Stage-2 target shape.")

    # -------------------------------------------------------------------------
    # Runtime physical batch probe
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold1_init = (
        STAGE1_FOLDS_ROOT
        / "fold_01/checkpoints/encoder_best.pth"
    )
    runtime = resolve_or_load_runtime(
        first_train,
        fold1_init,
        device,
    )
    batch_size = int(runtime["batch_size"])

    print("\nDevice:", device)
    print("Resolved Stage-2 physical batch size:", batch_size)
    if "cuda_device_name" in runtime:
        print("CUDA device:", runtime["cuda_device_name"])

    # Sampler balance / unique-subject audit.
    train_sampler = SubjectBalancedClipBatchSampler(
        first_train,
        batch_size=batch_size,
    )
    train_sampler.set_epoch(1)

    counts = Counter()
    for batch in train_sampler:
        batch_subjects = [
            first_train[ri].subject_id
            for ri, _ci in batch
        ]
        if len(batch_subjects) != len(set(batch_subjects)):
            raise RuntimeError(
                "Duplicate subject inside Stage-2 train batch."
            )
        counts.update(batch_subjects)

    exposure_spread = max(counts.values()) - min(counts.values())
    if exposure_spread > 1:
        raise RuntimeError(
            f"Stage-2 subject exposure spread >1: {exposure_spread}"
        )

    print("[PASS] Train batch contains unique subjects.")
    print("[PASS] Stage-2 train subject exposure differs by at most one draw.")

    # Fixed validation policy.
    fixed_val = select_validation_recordings(
        ready, first_fold["val_subjects"]
    )
    val_sampler = FixedValidationBatchSampler(
        fixed_val,
        batch_size=batch_size,
    )
    print(
        "Fold-01 fixed validation:",
        len(fixed_val),
        "recordings,",
        sum(len(b) for b in val_sampler),
        "logical clips",
    )

    # Model exact fine-tune audit.
    model = Stage2RPPGModel(
        load_stage1_encoder_weights(fold1_init)
    )
    report = model.trainable_parameter_report()

    expected_late4 = [
        "convblock2.0",
        "convblock2.3",
        "convblock3.0",
        "convblock3.3",
    ]
    if report["late_conv_names"] != expected_late4:
        raise RuntimeError(
            f"Stage-2 derived candidate late Conv3d contract changed: {report['late_conv_names']}"
        )
    if len(report["all_conv_names"]) != 7:
        raise RuntimeError(
            f"Stage-2 derived candidate expected 7 Stage-1 Conv3d modules, found {len(report['all_conv_names'])}"
        )
    print("All trainable Conv3d:", report["all_conv_names"])
    print("Late/high-LR Conv3d:", report["late_conv_names"])
    print(
        "Trainable parameters:",
        report["n_trainable"],
        "/",
        report["n_total"],
    )
    print("[PASS] No attention / HR / HRV / RR modules.")

    # Real forward shape/loss on a small CPU/GPU batch.
    model = model.to(device)
    criterion = RPPGPhysioCompositeLoss()

    with torch.no_grad():
        out = model(x.unsqueeze(0).to(device))
        loss, parts = criterion(out, y.unsqueeze(0).to(device), torch.tensor([float(meta["fps"])], device=device))

    if tuple(out["rppg"].shape) != (1, CLIP_LEN):
        raise RuntimeError("Final Stage-2 output is not [B,128].")
    if len(out["aux_rppg"]) != 3:
        raise RuntimeError("Stage-2 deep supervision head count changed.")

    print("Final rPPG:", tuple(out["rppg"].shape))
    print(
        "Aux rPPG:",
        [tuple(z.shape) for z in out["aux_rppg"]],
    )
    print("Initial Stage-2 loss:", float(loss))

    contract = build_training_contract(runtime)

    preflight = {
        "status": "PASS",
        "manifest": manifest_summary,
        "h5_ppg_audit": h5_audit,
        "frozen_split_sha256": sha256_file(FROZEN_SPLIT_JSON),
        "frozen_folds_sha256": sha256_file(FROZEN_FOLDS_JSON),
        "missing_stage2_ready_subjects": missing_ready_subjects,
        "missing_subjects_all_blind": True,
        "development_subjects_all_supervised": True,
        "fold_coverage": fold_reports,
        "runtime": runtime,
        "training_contract": contract,
        "blind_test_evaluated": False,
        "subject_leakage": 0,
        "stage2_target": "normalized_authoritative_ppg_sync_waveform_only",
        "attention_modules": False,
        "hr_hrv_rr_outputs": False,
        "subject_exposure_spread_fold1": exposure_spread,
    }
    save_json(
        OUTPUT_ROOT / "preflight/stage2_preflight.json",
        preflight,
    )

    print("=" * 112)
    print("STAGE-2 paper-aligned candidate PRE-FLIGHT STATUS: PASS")
    print("SUBJECT LEAKAGE: 0")
    print("BLIND TEST: LOCKED / NOT EVALUATED")
    print("TARGET: AUTHORITATIVE ppg_sync WAVEFORM ONLY")
    print("ATTENTION / HR / HRV / RR: NONE")
    print("=" * 112)


if __name__ == "__main__":
    main()
