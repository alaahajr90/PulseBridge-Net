from __future__ import annotations

import argparse
import logging
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_stage1 import *
from data_stage1 import (
    Stage1MCDH5Dataset,
    SubjectBalancedBatchSampler,
    ValidationUniqueSubjectBatchSampler,
    load_stage1_manifest,
    recordings_for_subjects,
    select_deterministic_validation_recordings,
)
from diagnostics_stage1 import (
    count_parameters,
    grad_global_norm,
    save_representation_plot,
    save_training_plots,
)
from io_utils import load_json, save_json, sha256_file, write_rows_csv
from model_stage1 import Encoder, ProjectionHead, contrastive_loss_components
from runtime_stage1 import resolve_or_load_runtime
from split_protocol import initialize_or_load_protocol, assert_no_subject_leakage
from training_contract import CONTRACT_JSON, build_training_contract


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=None, choices=range(1, K_FOLDS + 1))
    p.add_argument(
        "--skip-final",
        action="store_true",
        help="Run CV only; do not train final development encoder.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume compatible incomplete folds from last.pt and skip completed folds.",
    )
    return p.parse_args()


def setup_logger():
    log_dir = OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("stage1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler(path, encoding="utf-8")
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger, path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducibility is favored over peak throughput.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def worker_init_fn(worker_id):
    worker_seed = (SEED + worker_id) & 0xFFFFFFFF
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_train_loader(recordings, batch_size, epoch):
    ds = Stage1MCDH5Dataset(recordings, training=True)
    sampler = SubjectBalancedBatchSampler(
        recordings,
        batch_size=batch_size,
        seed=SEED,
    )
    sampler.set_epoch(epoch)

    return DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
        worker_init_fn=worker_init_fn,
    )


def make_val_loader(all_recordings, val_subjects, batch_size):
    refs = select_deterministic_validation_recordings(
        all_recordings, val_subjects
    )
    ds = Stage1MCDH5Dataset(refs, training=False)
    sampler = ValidationUniqueSubjectBatchSampler(refs, batch_size=batch_size)

    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
        worker_init_fn=worker_init_fn,
    )
    return loader, refs


def run_epoch(
    encoder,
    projector,
    loader,
    optimizer,
    device,
    training,
):
    encoder.train(training)
    projector.train(training)

    sums = {
        k: 0.0
        for k in [
            "loss",
            "contrastive",
            "temporal",
            "positive_cosine",
            "negative_cosine",
            "cosine_gap",
            "embedding_std",
            "grad_norm",
        ]
    }
    n_batches = 0
    n_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for x1, x2, _meta in tqdm(
            loader,
            desc="Train" if training else "Val",
            leave=False,
        ):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            components = contrastive_loss_components(
                projector(encoder(x1)),
                projector(encoder(x2)),
                TEMPERATURE,
                MARGIN,
                TEMPORAL_WEIGHT,
            )
            loss = components["total"]
            grad_norm = 0.0

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(projector.parameters()),
                    GRAD_CLIP_NORM,
                )
                grad_norm = grad_global_norm(
                    list(encoder.parameters()) + list(projector.parameters())
                )
                optimizer.step()

            batch_weight = int(x1.shape[0])
            sums["loss"] += float(loss.item()) * batch_weight
            for key in [
                "contrastive",
                "temporal",
                "positive_cosine",
                "negative_cosine",
                "cosine_gap",
                "embedding_std",
            ]:
                sums[key] += float(components[key].item()) * batch_weight
            sums["grad_norm"] += grad_norm * batch_weight

            n_batches += 1
            n_samples += batch_weight

    if n_samples == 0:
        raise RuntimeError("Empty epoch.")

    metrics = {k: v / n_samples for k, v in sums.items()}
    metrics["n_batches"] = n_batches
    metrics["n_samples"] = n_samples
    return metrics


def is_better(row, best):
    if best is None:
        return True
    if row["val_loss"] < best["val_loss"] - MIN_DELTA:
        return True
    return (
        abs(row["val_loss"] - best["val_loss"]) <= MIN_DELTA
        and row["val_cosine_gap"] > best["val_cosine_gap"]
    )


def checkpoint_contract():
    return load_json(CONTRACT_JSON)


def save_checkpoint(
    path,
    epoch,
    encoder,
    projector,
    optimizer,
    row,
    fold,
    best,
    patience_counter,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "stage1",
            "fold": fold,
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "projector": projector.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "selection_metrics": row,
            "best_so_far": best,
            "patience_counter": patience_counter,
            "selection_rule": {
                "primary": "minimum validation SSL loss",
                "secondary": "maximum validation cosine gap within MIN_DELTA tie",
                "ppg_used_for_selection": False,
            },
            "training_contract": checkpoint_contract(),
        },
        path,
    )


def verify_checkpoint_compatibility(payload):
    current = checkpoint_contract()
    saved = payload.get("training_contract")
    if saved != current:
        raise RuntimeError(
            "Checkpoint training contract does not match the current frozen "
            "manifest/protocol/code/runtime. Refusing unsafe resume."
        )


def maybe_resume_fold(
    last_path,
    encoder,
    projector,
    optimizer,
    history_path,
    logger,
):
    if not last_path.exists():
        return 1, None, 0, []

    payload = torch.load(last_path, map_location="cpu")
    verify_checkpoint_compatibility(payload)

    encoder.load_state_dict(payload["encoder"])
    projector.load_state_dict(payload["projector"])
    optimizer.load_state_dict(payload["optimizer"])

    start_epoch = int(payload["epoch"]) + 1
    best = payload.get("best_so_far")
    patience_counter = int(payload.get("patience_counter", 0))

    history = []
    if history_path.exists():
        import csv
        with open(history_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                parsed = {}
                for k, v in row.items():
                    try:
                        parsed[k] = float(v)
                    except Exception:
                        parsed[k] = v
                if "epoch" in parsed:
                    parsed["epoch"] = int(float(parsed["epoch"]))
                if "fold" in parsed:
                    parsed["fold"] = int(float(parsed["fold"]))
                history.append(parsed)

    logger.info(
        "Resuming compatible fold from epoch %d | patience=%d",
        start_epoch,
        patience_counter,
    )
    return start_epoch, best, patience_counter, history


def train_one_fold(
    fold_info,
    all_recordings,
    batch_size,
    device,
    logger,
    resume,
):
    fold = int(fold_info["fold"])
    fold_dir = OUTPUT_ROOT / "folds" / f"fold_{fold:02d}"
    ckpt_dir = fold_dir / "checkpoints"
    plot_dir = fold_dir / "plots"
    completed_path = fold_dir / "COMPLETED.json"
    history_path = fold_dir / "training_history.csv"

    fold_dir.mkdir(parents=True, exist_ok=True)

    if completed_path.exists():
        if not resume:
            raise RuntimeError(
                f"Fold {fold} already has COMPLETED.json. "
                "Use --resume to reuse compatible completed folds."
            )
        completed = load_json(completed_path)
        if completed.get("training_contract") != checkpoint_contract():
            raise RuntimeError(f"Completed fold {fold} contract mismatch.")
        logger.info("Fold %02d already complete; reusing frozen result.", fold)
        return completed["fold_summary"]

    train_refs = recordings_for_subjects(
        all_recordings, fold_info["train_subjects"]
    )
    val_loader, val_refs = make_val_loader(
        all_recordings,
        fold_info["val_subjects"],
        batch_size,
    )

    train_subject_set = {r.subject_id for r in train_refs}
    val_subject_set = {r.subject_id for r in val_refs}
    if train_subject_set & val_subject_set:
        raise RuntimeError(f"Recording-level leakage in fold {fold}.")

    logger.info("=" * 100)
    logger.info(
        "FOLD %02d/%02d | train subjects=%d | val subjects=%d | "
        "train READY recs=%d | val fixed recs=%d | batch=%d",
        fold,
        K_FOLDS,
        len(fold_info["train_subjects"]),
        len(fold_info["val_subjects"]),
        len(train_refs),
        len(val_refs),
        batch_size,
    )

    encoder = Encoder().to(device)
    projector = ProjectionHead(
        PROJ_INPUT_DIM,
        PROJ_HIDDEN_DIM,
        PROJ_OUTPUT_DIM,
        PROJ_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best = None
    patience_counter = 0
    history = []
    start_epoch = 1

    if resume and (ckpt_dir / "last.pt").exists():
        (
            start_epoch,
            best,
            patience_counter,
            history,
        ) = maybe_resume_fold(
            ckpt_dir / "last.pt",
            encoder,
            projector,
            optimizer,
            history_path,
            logger,
        )

    if start_epoch > EPOCHS:
        raise RuntimeError(
            f"Fold {fold} last checkpoint exceeds configured EPOCHS but "
            "has no compatible COMPLETED marker."
        )

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loader = make_train_loader(
            train_refs,
            batch_size=batch_size,
            epoch=epoch,
        )

        train_metrics = run_epoch(
            encoder, projector, train_loader, optimizer, device, True
        )
        val_metrics = run_epoch(
            encoder, projector, val_loader, None, device, False
        )

        row = {
            "fold": fold,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        write_rows_csv(history_path, history)

        logger.info(
            "Fold %02d | Epoch %03d/%03d | "
            "train=%.6f | val=%.6f | val_gap=%.4f | "
            "emb_std=%.6f | grad=%.4f | samples=%d/%d",
            fold,
            epoch,
            EPOCHS,
            row["train_loss"],
            row["val_loss"],
            row["val_cosine_gap"],
            row["val_embedding_std"],
            row["train_grad_norm"],
            int(row["train_n_samples"]),
            int(row["val_n_samples"]),
        )

        if row["val_embedding_std"] < MIN_ACCEPTABLE_EMBED_STD:
            logger.warning(
                "Possible representation collapse: val_embedding_std=%.8f",
                row["val_embedding_std"],
            )

        improved = is_better(row, best)
        if improved:
            best = dict(row)
            patience_counter = 0
            save_checkpoint(
                ckpt_dir / "best.pt",
                epoch,
                encoder,
                projector,
                optimizer,
                row,
                fold,
                best,
                patience_counter,
            )
            torch.save(encoder.state_dict(), ckpt_dir / "encoder_best.pth")
            logger.info(
                "  -> NEW BEST fold %02d checkpoint at epoch %d",
                fold,
                epoch,
            )
        else:
            patience_counter += 1

        # Save last after best/patience state is updated.
        save_checkpoint(
            ckpt_dir / "last.pt",
            epoch,
            encoder,
            projector,
            optimizer,
            row,
            fold,
            best,
            patience_counter,
        )

        if SAVE_PLOTS_EVERY_EPOCH:
            save_training_plots(
                history,
                plot_dir / "loss_curves.png",
                f"Stage 1 Fold {fold:02d}",
            )
            save_representation_plot(
                history,
                plot_dir / "representation_diagnostics.png",
                f"Stage 1 Fold {fold:02d}",
            )

        if patience_counter >= PATIENCE:
            logger.info(
                "Early stopping fold %02d at epoch %d | best epoch=%d val=%.6f",
                fold,
                epoch,
                best["epoch"],
                best["val_loss"],
            )
            break

    if best is None:
        raise RuntimeError(f"Fold {fold} produced no best checkpoint.")

    summary = {
        "fold": fold,
        "n_train_subjects": len(fold_info["train_subjects"]),
        "n_val_subjects": len(fold_info["val_subjects"]),
        "n_train_ready_recordings": len(train_refs),
        "n_val_fixed_recordings": len(val_refs),
        "physical_batch_size": batch_size,
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "best_val_cosine_gap": float(best["val_cosine_gap"]),
        "best_val_embedding_std": float(best["val_embedding_std"]),
        "best_checkpoint": str(ckpt_dir / "best.pt"),
        "encoder_checkpoint": str(ckpt_dir / "encoder_best.pth"),
        "ppg_used_for_training_or_selection": False,
        "blind_test_used": False,
    }

    save_json(fold_dir / "fold_summary.json", summary)
    save_json(
        completed_path,
        {
            "training_contract": checkpoint_contract(),
            "fold_summary": summary,
        },
    )
    return summary


def train_final_development(
    all_recordings,
    protocol,
    best_epochs,
    batch_size,
    device,
    logger,
):
    final_epochs = int(round(statistics.median(best_epochs)))
    final_dir = OUTPUT_ROOT / "final_model"
    ckpt_dir = final_dir / "checkpoints"
    final_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dev_refs = recordings_for_subjects(
        all_recordings, protocol["development_subjects"]
    )
    blind_subjects = set(protocol["blind_test_subjects"])

    if any(r.subject_id in blind_subjects for r in dev_refs):
        raise RuntimeError("Blind subject entered final Stage-1 development training.")

    logger.info("=" * 100)
    logger.info(
        "FINAL DEVELOPMENT TRAINING | subjects=%d | READY recs=%d | "
        "fixed epochs=%d | batch=%d | blind remains LOCKED",
        len(protocol["development_subjects"]),
        len(dev_refs),
        final_epochs,
        batch_size,
    )

    encoder = Encoder().to(device)
    projector = ProjectionHead(
        PROJ_INPUT_DIM,
        PROJ_HIDDEN_DIM,
        PROJ_OUTPUT_DIM,
        PROJ_DROPOUT,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    for epoch in range(1, final_epochs + 1):
        loader = make_train_loader(
            dev_refs,
            batch_size=batch_size,
            epoch=100000 + epoch,
        )
        metrics = run_epoch(
            encoder, projector, loader, optimizer, device, True
        )
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in metrics.items()},
        }
        history.append(row)
        write_rows_csv(final_dir / "training_history.csv", history)

        logger.info(
            "FINAL DEV | Epoch %03d/%03d | train=%.6f | "
            "gap=%.4f | emb_std=%.6f | samples=%d",
            epoch,
            final_epochs,
            row["train_loss"],
            row["train_cosine_gap"],
            row["train_embedding_std"],
            int(row["train_n_samples"]),
        )

    final_checkpoint = ckpt_dir / "stage1_final_development.pt"
    torch.save(
        {
            "stage": "stage1_final_development",
            "epochs": final_epochs,
            "encoder": encoder.state_dict(),
            "projector": projector.state_dict(),
            "cv_best_epochs": list(map(int, best_epochs)),
            "blind_test_used_for_training": False,
            "blind_test_evaluated": False,
            "ppg_used": False,
            "training_contract": checkpoint_contract(),
        },
        final_checkpoint,
    )

    encoder_path = final_dir / "stage1_encoder_for_stage2.pth"
    torch.save(encoder.state_dict(), encoder_path)

    blind_lock = {
        "status": "LOCKED_NOT_EVALUATED_IN_STAGE1",
        "n_blind_subjects": len(protocol["blind_test_subjects"]),
        "blind_subjects_sha256": sha256_file(FROZEN_SPLIT_JSON),
        "metrics_created": False,
        "reason": (
            "Preserve one untouched end-to-end blind subject set for downstream "
            "Stage 2 / Stage 3 final evaluation."
        ),
    }
    save_json(final_dir / "blind_test_LOCKED.json", blind_lock)

    logger.info("Stage-2 initialization encoder: %s", encoder_path)
    logger.info("Blind test remains LOCKED; no Stage-1 blind metrics were computed.")

    return {
        "final_epochs": final_epochs,
        "final_checkpoint": str(final_checkpoint),
        "stage2_encoder": str(encoder_path),
        "blind_test_status": "LOCKED_NOT_EVALUATED_IN_STAGE1",
        "ppg_used": False,
    }


def main():
    args = parse_args()
    seed_everything(SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger, log_path = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("PulseBridge-Net Stage 1 FINAL")
    logger.info("Device=%s | Encoder params=%d", device, count_parameters(Encoder()))

    # Frozen data + subject protocol.
    accepted, rejected, qc_summary = load_stage1_manifest()
    protocol, folds, protocol_manifest = initialize_or_load_protocol()
    assert_no_subject_leakage(protocol, folds)

    # Preflight must already have frozen the runtime batch. If the user invokes
    # training directly, resolve it here with the same probe logic.
    runtime = resolve_or_load_runtime(accepted, device)
    batch_size = int(runtime["batch_size"])
    contract = build_training_contract(protocol_manifest, runtime)

    logger.info(
        "FROZEN QC MANIFEST | READY=%d HOLD=%d SHA=%s",
        len(accepted),
        len(rejected),
        qc_summary["manifest_sha256"],
    )
    logger.info(
        "FROZEN BLIND: %d/%d subjects | NOT EVALUATED IN STAGE 1",
        protocol["n_blind_test_subjects"],
        protocol["n_total_subjects"],
    )
    logger.info(
        "DEVELOPMENT: %d subjects | K=%d | physical batch=%d",
        protocol["n_development_subjects"],
        K_FOLDS,
        batch_size,
    )
    logger.info("Split SHA256=%s", protocol_manifest["frozen_split_sha256"])
    logger.info("Folds SHA256=%s", protocol_manifest["frozen_folds_sha256"])

    target_folds = (
        folds["folds"]
        if args.fold is None
        else [f for f in folds["folds"] if int(f["fold"]) == args.fold]
    )

    summaries = [
        train_one_fold(
            fold_info,
            accepted,
            batch_size,
            device,
            logger,
            resume=args.resume,
        )
        for fold_info in target_folds
    ]

    if args.fold is not None:
        logger.info("Single-fold run complete; final model was not trained.")
        return

    write_rows_csv(
        OUTPUT_ROOT / "cross_validation" / "cv_summary.csv",
        summaries,
    )
    save_json(
        OUTPUT_ROOT / "cross_validation" / "cv_summary.json",
        summaries,
    )

    losses = [s["best_val_loss"] for s in summaries]
    gaps = [s["best_val_cosine_gap"] for s in summaries]
    epochs = [s["best_epoch"] for s in summaries]

    aggregate = {
        "k": K_FOLDS,
        "best_val_loss_mean": float(np.mean(losses)),
        "best_val_loss_std": float(np.std(losses, ddof=1)),
        "best_val_cosine_gap_mean": float(np.mean(gaps)),
        "best_val_cosine_gap_std": float(np.std(gaps, ddof=1)),
        "best_epochs": list(map(int, epochs)),
        "median_best_epoch": int(round(statistics.median(epochs))),
        "blind_test_used": False,
        "ppg_used": False,
        "training_contract": contract,
    }
    save_json(
        OUTPUT_ROOT / "cross_validation" / "cv_aggregate.json",
        aggregate,
    )

    logger.info(
        "10-FOLD CV COMPLETE | val loss %.6f ± %.6f | gap %.4f ± %.4f",
        aggregate["best_val_loss_mean"],
        aggregate["best_val_loss_std"],
        aggregate["best_val_cosine_gap_mean"],
        aggregate["best_val_cosine_gap_std"],
    )
    logger.info(
        "Best epochs=%s | median=%d",
        epochs,
        aggregate["median_best_epoch"],
    )

    if not args.skip_final:
        final_summary = train_final_development(
            accepted,
            protocol,
            epochs,
            batch_size,
            device,
            logger,
        )
        save_json(
            OUTPUT_ROOT / "final_model" / "final_summary.json",
            final_summary,
        )

    logger.info("Complete log: %s", log_path)


if __name__ == "__main__":
    main()
