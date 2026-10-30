from __future__ import annotations
import gc
import torch
from torch.utils.data import DataLoader

from config_stage2 import *
from data_stage2 import Stage2RPPGDataset, SubjectBalancedClipBatchSampler
from io_utils import load_json, save_json
from losses_stage2 import RPPGPhysioCompositeLoss
from model_stage2 import Stage2RPPGModel, load_stage1_encoder_weights

RUNTIME_JSON = OUTPUT_ROOT / "preflight/resolved_runtime.json"


def make_optimizer(model):
    early, late, heads = model.parameter_groups()
    groups = []
    if early:
        groups.append({"params": early, "lr": LR_ENCODER_EARLY, "name": "encoder_early"})
    if late:
        groups.append({"params": late, "lr": LR_ENCODER_LATE, "name": "encoder_late"})
    groups.append({"params": heads, "lr": LR_DECODER, "name": "decoder_heads"})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def _probe(recordings, stage1_init, batch_size, device):
    ds = Stage2RPPGDataset(recordings)
    sampler = SubjectBalancedClipBatchSampler(recordings, batch_size=batch_size)
    sampler.set_epoch(0)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    model = Stage2RPPGModel(load_stage1_encoder_weights(stage1_init)).to(device)
    optimizer = make_optimizer(model)
    criterion = RPPGPhysioCompositeLoss()

    x, y, meta = next(iter(loader))
    x, y = x.to(device), y.to(device)
    fps = torch.as_tensor(meta["fps"], dtype=torch.float32, device=device)

    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    loss, parts = criterion(out, y, fps)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite paper-aligned probe loss: {loss}")
    loss.backward()
    optimizer.step()

    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    return {
        "batch_size": batch_size,
        "probe_loss": float(loss.detach().cpu()),
        "probe_corr_loss": float(parts["corr_loss"].detach().cpu()),
        "probe_std_loss": float(parts["std_loss"].detach().cpu()),
        "peak_cuda_memory_bytes": peak,
    }


def resolve_or_load_runtime(recordings, stage1_init, device, reprobe=False):
    if RUNTIME_JSON.exists() and not reprobe:
        payload = load_json(RUNTIME_JSON)
        if payload.get("runtime_contract_version") == "Stage2_PAPER_ALIGNED_DERIVED":
            return payload

    if device.type != "cuda":
        payload = {
            "runtime_contract_version": "Stage2_PAPER_ALIGNED_DERIVED",
            "device": str(device),
            "batch_size": DEFAULT_CPU_BATCH_SIZE,
            "selection": "fixed CPU fallback",
            "probe_results": [],
        }
        save_json(RUNTIME_JSON, payload)
        return payload

    results, chosen = [], None
    for batch_size in BATCH_SIZE_CANDIDATES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gc.collect()
        try:
            result = _probe(recordings, stage1_init, batch_size, device)
            result["status"] = "PASS"
            results.append(result)
            chosen = batch_size
            break
        except torch.cuda.OutOfMemoryError as exc:
            results.append({"batch_size": batch_size, "status": "CUDA_OOM", "message": str(exc)})
            torch.cuda.empty_cache()
            gc.collect()

    if chosen is None:
        raise RuntimeError("No Stage-2 paper-aligned batch candidate fits the GPU.")

    props = torch.cuda.get_device_properties(device)
    payload = {
        "runtime_contract_version": "Stage2_PAPER_ALIGNED_DERIVED",
        "device": str(device),
        "cuda_device_name": props.name,
        "cuda_total_memory_bytes": int(props.total_memory),
        "batch_size": int(chosen),
        "selection": "largest paper-aligned candidate passing real forward+backward",
        "probe_results": results,
    }
    save_json(RUNTIME_JSON, payload)
    return payload
