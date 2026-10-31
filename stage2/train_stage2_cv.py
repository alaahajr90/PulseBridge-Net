from __future__ import annotations

import argparse, csv, gc, logging, random, statistics, sys
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_stage2 import *
from data_stage2 import (
    FixedValidationBatchSampler, Stage2RPPGDataset,
    SubjectBalancedClipBatchSampler, load_stage2_manifest,
    recordings_for_subjects, select_validation_recordings,
)
from diagnostics_stage2 import save_loss_plot, save_metric_plot, save_waveform_examples
from io_utils import atomic_torch_save, load_json, save_json, sha256_file, torch_load_full, write_rows_csv
from losses_stage2 import RPPGPhysioCompositeLoss
from metrics_stage2 import SubjectMetricAccumulator, spectral_correlation_per_sample, waveform_metrics_per_sample
from model_stage2 import Stage2RPPGModel, load_stage1_encoder_weights
from protocol_stage2 import load_stage1_protocol
from runtime_stage2 import resolve_or_load_runtime, make_optimizer
from training_contract import CONTRACT_JSON, build_training_contract


def parse_args():
    p = argparse.ArgumentParser("PulseBridge-Net Stage 2 v4 MorphologyPreserving")
    p.add_argument("--fold", type=int, choices=range(1, K_FOLDS + 1), default=None)
    p.add_argument("--skip-final", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def worker_init_fn(worker_id):
    s = (SEED + worker_id) & 0xFFFFFFFF
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def setup_logger():
    d = OUTPUT_ROOT / "logs"; d.mkdir(parents=True, exist_ok=True)
    path = d / f"stage2_v4_morph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log = logging.getLogger("stage2_v4_morph"); log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt); log.addHandler(h)
    return log, path


def loader_kwargs():
    kw = {
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY and torch.cuda.is_available(),
        "persistent_workers": PERSISTENT_WORKERS and NUM_WORKERS > 0,
        "worker_init_fn": worker_init_fn,
    }
    if NUM_WORKERS > 0:
        kw["prefetch_factor"] = PREFETCH_FACTOR
    return kw


def make_train_loader(recordings, batch_size):
    ds = Stage2RPPGDataset(recordings)
    sampler = SubjectBalancedClipBatchSampler(recordings, batch_size=batch_size, seed=SEED)
    return DataLoader(ds, batch_sampler=sampler, **loader_kwargs()), sampler


def make_val_loader(all_recordings, subjects, batch_size):
    selected = select_validation_recordings(all_recordings, subjects)
    ds = Stage2RPPGDataset(selected)
    sampler = FixedValidationBatchSampler(selected, batch_size=batch_size)
    return DataLoader(ds, batch_sampler=sampler, **loader_kwargs()), selected


def _meta_list(meta, key):
    value = meta[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value)


def run_epoch(model, loader, criterion, optimizer, device, training, collect_examples=False):
    model.train(training)
    acc = SubjectMetricAccumulator()
    examples = []
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        bar = tqdm(loader, desc="Train" if training else "Val", leave=False)
        for x, y, meta in bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            fps = torch.as_tensor(meta["fps"], dtype=torch.float32, device=device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            out = model(x)
            loss, parts = criterion(out, y, fps)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Stage-2 v4-MORPH loss: {loss}")

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], GRAD_CLIP_NORM
                )
                optimizer.step()

            wf = waveform_metrics_per_sample(out["rppg"], y, fps)
            spec = spectral_correlation_per_sample(out["rppg"], y, fps, PULSE_BAND_HZ, PSD_NFFT)
            per = {
                "total_loss": parts["total_per_sample"].detach().cpu().numpy(),
                "final_loss": parts["final_per_sample"].detach().cpu().numpy(),
                "corr_loss": parts["corr_loss_per_sample"].detach().cpu().numpy(),
                "wave_loss": parts["wave_per_sample"].detach().cpu().numpy(),
                "derivative_loss": parts["derivative_per_sample"].detach().cpu().numpy(),
                "spectrum_loss": parts["spectrum_per_sample"].detach().cpu().numpy(),
                "band_loss": parts["band_per_sample"].detach().cpu().numpy(),
                "std_loss": parts["std_loss_per_sample"].detach().cpu().numpy(),
                "aux_mean": parts["aux_mean_per_sample"].detach().cpu().numpy(),
                "baseline_final_loss": parts["baseline_final_per_sample"].detach().cpu().numpy(),
                "morphology_loss": parts["morphology_per_sample"].detach().cpu().numpy(),
                "morph_d1_loss": parts["morph_d1_per_sample"].detach().cpu().numpy(),
                "morph_d2_loss": parts["morph_d2_per_sample"].detach().cpu().numpy(),
                "morph_multiscale_loss": parts["morph_multiscale_per_sample"].detach().cpu().numpy(),
                **wf,
                "spectral_corr": spec,
            }

            subjects = _meta_list(meta, "subject_id")
            paths = _meta_list(meta, "h5_path")
            cameras = _meta_list(meta, "camera")
            states = _meta_list(meta, "state")
            acc.add(subjects, paths, per, cameras=cameras, states=states)

            bar.set_postfix(
                loss=f"{float(loss.item()):.3f}",
                r=f"{float(np.nanmean(wf['pearson_r'])):.3f}",
                std=f"{float(np.nanmean(wf['std_ratio'])):.2f}",
            )

            if collect_examples and len(examples) < N_WAVEFORM_PLOTS:
                pp = out["rppg"].detach().cpu().numpy()
                yy = y.detach().cpu().numpy()
                starts = _meta_list(meta, "start_idx")
                for i in range(min(len(pp), N_WAVEFORM_PLOTS - len(examples))):
                    examples.append({
                        "pred": pp[i].tolist(),
                        "target": yy[i].tolist(),
                        "title": f"{subjects[i]} {states[i]} {cameras[i]} start={int(starts[i])}",
                    })

    return acc.finalize(), examples, acc.grouped_finalize()


def current_contract():
    return load_json(CONTRACT_JSON)


def save_checkpoint(path, model, optimizer, scheduler, epoch, row, fold, stage1_init, best, stale):
    atomic_torch_save({
        "stage": "stage2_v4_morphology_preserving_rppg_waveform_only",
        "fold": fold,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metrics": row,
        "best_so_far": best,
        "patience_counter": int(stale),
        "stage1_init": str(stage1_init),
        "stage1_init_sha256": sha256_file(stage1_init),
        "outputs": ["normalized_rppg_waveform"],
        "attention_modules": False,
        "blind_test_evaluated": False,
        "training_contract": current_contract(),
    }, path)


def verify_checkpoint_contract(payload):
    if payload.get("training_contract") != current_contract():
        raise RuntimeError("Unsafe resume blocked: Stage-2 v4-MORPH training contract mismatch.")


def load_history(path):
    if not path.exists(): return []
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                try: parsed[k] = float(v)
                except Exception: parsed[k] = v
            if "epoch" in parsed: parsed["epoch"] = int(float(parsed["epoch"]))
            if "fold" in parsed: parsed["fold"] = int(float(parsed["fold"]))
            rows.append(parsed)
    return rows


def is_better(row, best):
    if best is None: return True
    if row["val_total_loss"] < best["val_total_loss"] - MIN_DELTA: return True
    if abs(row["val_total_loss"] - best["val_total_loss"]) <= MIN_DELTA:
        return row["val_pearson_r"] > best["val_pearson_r"]
    return False


def train_one_fold(fold_info, all_recordings, batch_size, device, log, resume):
    fold = int(fold_info["fold"])
    directory = OUTPUT_ROOT / "folds" / f"fold_{fold:02d}"
    ck = directory / "checkpoints"; plots = directory / "plots"
    hist_path = directory / "training_history.csv"
    done_path = directory / "COMPLETED.json"
    directory.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)

    if done_path.exists():
        done = load_json(done_path)
        if not resume or done.get("training_contract") != current_contract():
            raise RuntimeError(f"Existing Fold {fold} is incompatible with Stage-2 v4 MorphologyPreserving.")
        log.info("Stage-2 v4-MORPH Fold %02d already complete; reusing.", fold)
        return done["fold_summary"]

    train_refs = recordings_for_subjects(all_recordings, fold_info["train_subjects"])
    val_loader, val_selected = make_val_loader(all_recordings, fold_info["val_subjects"], batch_size)
    if {r.subject_id for r in train_refs} & {r.subject_id for r in val_selected}:
        raise RuntimeError(f"Leakage in Stage-2 v4-MORPH fold {fold}")

    init = STAGE1_FOLDS_ROOT / f"fold_{fold:02d}/checkpoints/encoder_best.pth"
    model = Stage2RPPGModel(load_stage1_encoder_weights(init)).to(device)
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE, min_lr=LR_MIN
    )
    criterion = RPPGPhysioCompositeLoss()
    train_loader, train_sampler = make_train_loader(train_refs, batch_size)

    report = model.trainable_parameter_report()
    save_json(directory/"trainable_parameter_report.json", report)

    log.info("="*108)
    log.info(
        "STAGE-2 v4 MORPHOLOGY-PRESERVING FOLD %02d/%02d | train subjects=%d | val=%d | train recs=%d | val fixed=%d | batch=%d",
        fold, K_FOLDS, len(fold_info["train_subjects"]), len(fold_info["val_subjects"]),
        len(train_refs), len(val_selected), batch_size
    )
    log.info("Stage-1 fold init: %s | SHA=%s", init, sha256_file(init))
    log.info("Early Conv3d LR=%g | late Conv3d LR=%g | decoder LR=%g",
             LR_ENCODER_EARLY, LR_ENCODER_LATE, LR_DECODER)
    log.info("Temporal decoder dilations=%s | composite anti-collapse loss active", DECODER_DILATIONS)

    start, best, stale, history = 1, None, 0, []
    if resume and (ck/"last.pt").exists():
        payload = torch_load_full(ck/"last.pt", map_location="cpu")
        verify_checkpoint_contract(payload)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler"): scheduler.load_state_dict(payload["scheduler"])
        start = int(payload["epoch"]) + 1
        best = payload.get("best_so_far")
        stale = int(payload.get("patience_counter", 0))
        history = load_history(hist_path)
        log.info("Resuming compatible Stage-2 v4-MORPH fold from epoch %d | patience=%d", start, stale)

    for epoch in range(start, EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        tr, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, True, False)
        va, examples, groups = run_epoch(model, val_loader, criterion, None, device, False, True)

        row = {
            "fold": fold, "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
        }
        history.append(row); write_rows_csv(hist_path, history)
        scheduler.step(row["val_total_loss"])

        lrs = [g["lr"] for g in optimizer.param_groups]
        log.info(
            "Fold %02d | Epoch %03d/%03d | loss %.4f/%.4f | morph %.4f | d1shape %.4f | d2shape %.4f | "
            "r %.4f | specR %.4f | RMSE %.4f | predStd %.3f | stdRatio %.3f | bestLagR %.4f | lag %.3fs | LR=%s",
            fold, epoch, EPOCHS, row["train_total_loss"], row["val_total_loss"],
            row["val_morphology_loss"], row["val_morph_d1_loss"], row["val_morph_d2_loss"],
            row["val_pearson_r"], row["val_spectral_corr"], row["val_rmse"],
            row["val_pred_std"], row["val_std_ratio"], row["val_best_lag_r"],
            row["val_best_lag_seconds"], ",".join(f"{x:.1e}" for x in lrs)
        )

        improved = is_better(row, best)
        if improved:
            best = dict(row); stale = 0
            save_checkpoint(ck/"best.pt", model, optimizer, scheduler, epoch, row, fold, init, best, stale)
            atomic_torch_save(model.state_dict(), ck/"stage2_rppg_best.pth")
            save_json(directory/"best_group_diagnostics.json", groups)
            save_waveform_examples(examples, plots/"best_waveforms.png", f"Stage 2 v4 MorphologyPreserving Fold {fold:02d}")
            log.info("  -> NEW BEST v4-MORPH fold %02d at epoch %d | r=%.4f stdRatio=%.3f",
                     fold, epoch, row["val_pearson_r"], row["val_std_ratio"])
        else:
            stale += 1

        save_checkpoint(ck/"last.pt", model, optimizer, scheduler, epoch, row, fold, init, best, stale)
        if SAVE_PLOTS_EVERY_EPOCH:
            save_loss_plot(history, plots/"loss_curves.png", f"Stage 2 v4 MorphologyPreserving Fold {fold:02d}")
            save_metric_plot(history, plots/"waveform_metrics.png", f"Stage 2 v4 MorphologyPreserving Fold {fold:02d}")

        if stale >= PATIENCE:
            log.info("Early stopping v4-MORPH fold %02d at epoch %d | best=%d val=%.4f r=%.4f",
                     fold, epoch, int(best["epoch"]), best["val_total_loss"], best["val_pearson_r"])
            break

    summary = {
        "fold": fold,
        "best_epoch": int(best["epoch"]),
        "best_val_total_loss": float(best["val_total_loss"]),
        "best_val_rmse": float(best["val_rmse"]),
        "best_val_pearson_r": float(best["val_pearson_r"]),
        "best_val_spectral_corr": float(best["val_spectral_corr"]),
        "best_val_pred_std": float(best["val_pred_std"]),
        "best_val_std_ratio": float(best["val_std_ratio"]),
        "best_val_morphology_loss": float(best["val_morphology_loss"]),
        "best_val_morph_d1_loss": float(best["val_morph_d1_loss"]),
        "best_val_morph_d2_loss": float(best["val_morph_d2_loss"]),
        "best_val_morph_multiscale_loss": float(best["val_morph_multiscale_loss"]),
        "best_val_best_lag_r": float(best["val_best_lag_r"]),
        "best_val_best_lag_seconds": float(best["val_best_lag_seconds"]),
        "stage1_init": str(init),
        "blind_test_used": False,
    }
    save_json(directory/"fold_summary.json", summary)
    save_json(done_path, {"training_contract": current_contract(), "fold_summary": summary})
    del train_loader, val_loader; gc.collect()
    return summary


def final_fit(all_recordings, protocol, best_epochs, batch_size, device, log, resume):
    epochs = int(round(statistics.median(best_epochs)))
    directory = OUTPUT_ROOT/"final_model"; ck = directory/"checkpoints"
    hist_path = directory/"training_history.csv"; done_path = directory/"COMPLETED.json"
    directory.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)

    if done_path.exists():
        done = load_json(done_path)
        if done.get("training_contract") != current_contract():
            raise RuntimeError("Final v3 result contract mismatch.")
        return done["final_summary"]

    dev_refs = recordings_for_subjects(all_recordings, protocol["development_subjects"])
    blind = set(map(str, protocol["blind_test_subjects"]))
    if any(r.subject_id in blind for r in dev_refs):
        raise RuntimeError("Blind subject entered final Stage-2 v4-MORPH fit.")

    model = Stage2RPPGModel(load_stage1_encoder_weights(STAGE1_FINAL_ENCODER)).to(device)
    optimizer = make_optimizer(model)
    criterion = RPPGPhysioCompositeLoss()
    loader, sampler = make_train_loader(dev_refs, batch_size)

    start, history = 1, []
    if resume and (ck/"last.pt").exists():
        payload = torch_load_full(ck/"last.pt", map_location="cpu")
        verify_checkpoint_contract(payload)
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        start = int(payload["epoch"]) + 1; history = load_history(hist_path)
        log.info("Resuming FINAL Stage-2 v4 MorphologyPreserving from epoch %d", start)

    log.info("="*108)
    log.info("FINAL Stage-2 v4 MorphologyPreserving DEVELOPMENT | subjects=540 | recs=%d | epochs=%d | blind LOCKED",
             len(dev_refs), epochs)

    for epoch in range(start, epochs+1):
        sampler.set_epoch(100000 + epoch)
        tr, _, _ = run_epoch(model, loader, criterion, optimizer, device, True, False)
        row = {"epoch": epoch, **{f"train_{k}": v for k,v in tr.items()}}
        history.append(row); write_rows_csv(hist_path, history)
        log.info("FINAL v4-MORPH | %03d/%03d | loss=%.4f r=%.4f RMSE=%.4f stdRatio=%.3f",
                 epoch, epochs, row["train_total_loss"], row["train_pearson_r"],
                 row["train_rmse"], row["train_std_ratio"])
        save_checkpoint(ck/"last.pt", model, optimizer, None, epoch,
                        {"final_dev_epochs": epochs, **row}, "final",
                        STAGE1_FINAL_ENCODER, None, 0)

    official = directory/"stage2_rppg_for_stage3.pt"
    atomic_torch_save({
        "stage": "stage2_v4_morphology_preserving_development_rppg_only",
        "epoch": epochs,
        "model": model.state_dict(),
        "stage1_init": str(STAGE1_FINAL_ENCODER),
        "stage1_init_sha256": sha256_file(STAGE1_FINAL_ENCODER),
        "outputs": ["normalized_rppg_waveform"],
        "target_training_source": "authoritative GOOD ppg_sync only",
        "blind_test_evaluated": False,
        "training_contract": current_contract(),
    }, official)

    save_json(directory/"blind_test_LOCKED.json", {
        "status": "LOCKED_NOT_EVALUATED_IN_STAGE2",
        "n_frozen_blind_subjects": len(protocol["blind_test_subjects"]),
        "zero_stage2_supervision_blind_subjects": list(EXPECTED_NO_STAGE2_READY_SUBJECTS),
        "metrics_created": False,
    })
    summary = {
        "final_epochs": epochs,
        "official_stage3_model": str(official),
        "n_development_subjects": 540,
        "n_development_ready_recordings": len(dev_refs),
        "blind_test_status": "LOCKED_NOT_EVALUATED_IN_STAGE2",
    }
    save_json(directory/"final_summary.json", summary)
    save_json(done_path, {"training_contract": current_contract(), "final_summary": summary})
    return summary


def main():
    args = parse_args(); seed_everything()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log, log_path = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol, folds, _ = load_stage1_protocol()
    recs, hold, _ = load_stage2_manifest()

    fold1_train = recordings_for_subjects(recs, folds["folds"][0]["train_subjects"])
    fold1_init = STAGE1_FOLDS_ROOT/"fold_01/checkpoints/encoder_best.pth"
    runtime = resolve_or_load_runtime(fold1_train, fold1_init, device)
    batch = int(runtime["batch_size"])
    contract = build_training_contract(runtime)

    log.info("PulseBridge-Net Stage 2 v4 MorphologyPreserving — PHYSIOLOGICAL rPPG WAVEFORM")
    log.info("Device=%s | READY=%d HOLD=%d | batch=%d | epochs=%d", device, len(recs), len(hold), batch, EPOCHS)
    log.info("Original v3 loss preserved: corr=%g wave=%g deriv=%g spectrum=%g band=%g std=%g aux=%g",
             W_CORR, W_WAVE, W_DERIVATIVE, W_SPECTRUM, W_BAND, W_STD, AUX_WEIGHT)
    log.info("ONLY v4 additions: morph_d1=%g morph_d2=%g multiscale=%g lags=%s",
             W_MORPH_D1, W_MORPH_D2, W_MORPH_MULTISCALE, MORPH_MULTISCALE_LAGS)
    log.info("Blind test LOCKED. No attention / HR / HRV / RR.")

    targets = folds["folds"] if args.fold is None else [f for f in folds["folds"] if int(f["fold"]) == args.fold]
    summaries = [train_one_fold(f, recs, batch, device, log, args.resume) for f in targets]

    if args.fold is not None:
        return

    write_rows_csv(OUTPUT_ROOT/"cross_validation/cv_summary.csv", summaries)
    save_json(OUTPUT_ROOT/"cross_validation/cv_summary.json", summaries)
    epochs = [s["best_epoch"] for s in summaries]
    aggregate = {
        "k": 10,
        "best_epochs": epochs,
        "median_best_epoch": int(round(statistics.median(epochs))),
        "val_rmse_mean": float(np.mean([s["best_val_rmse"] for s in summaries])),
        "val_rmse_std": float(np.std([s["best_val_rmse"] for s in summaries], ddof=1)),
        "val_pearson_r_mean": float(np.mean([s["best_val_pearson_r"] for s in summaries])),
        "val_pearson_r_std": float(np.std([s["best_val_pearson_r"] for s in summaries], ddof=1)),
        "val_spectral_corr_mean": float(np.mean([s["best_val_spectral_corr"] for s in summaries])),
        "val_std_ratio_mean": float(np.mean([s["best_val_std_ratio"] for s in summaries])),
        "val_morphology_loss_mean": float(np.mean([s["best_val_morphology_loss"] for s in summaries])),
        "val_morph_d1_loss_mean": float(np.mean([s["best_val_morph_d1_loss"] for s in summaries])),
        "val_morph_d2_loss_mean": float(np.mean([s["best_val_morph_d2_loss"] for s in summaries])),
        "val_morph_multiscale_loss_mean": float(np.mean([s["best_val_morph_multiscale_loss"] for s in summaries])),
        "val_best_lag_r_mean": float(np.mean([s["best_val_best_lag_r"] for s in summaries])),
        "blind_test_used": False,
        "training_contract": contract,
    }
    save_json(OUTPUT_ROOT/"cross_validation/cv_aggregate.json", aggregate)
    log.info("10-FOLD v4-MORPH COMPLETE | r=%.4f±%.4f RMSE=%.4f±%.4f specR=%.4f stdRatio=%.3f bestLagR=%.4f",
             aggregate["val_pearson_r_mean"], aggregate["val_pearson_r_std"],
             aggregate["val_rmse_mean"], aggregate["val_rmse_std"],
             aggregate["val_spectral_corr_mean"], aggregate["val_std_ratio_mean"],
             aggregate["val_best_lag_r_mean"])

    if not args.skip_final:
        final_fit(recs, protocol, epochs, batch, device, log, args.resume)
    log.info("Complete log: %s", log_path)


if __name__ == "__main__":
    main()
