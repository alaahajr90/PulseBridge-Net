from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch.utils.data import DataLoader

from config_stage1 import (
    BATCH_SIZE_CANDIDATES,
    DEFAULT_CPU_BATCH_SIZE,
    OUTPUT_ROOT,
    PROJ_DROPOUT,
    PROJ_HIDDEN_DIM,
    PROJ_INPUT_DIM,
    PROJ_OUTPUT_DIM,
    TEMPERATURE,
    MARGIN,
    TEMPORAL_WEIGHT,
)
from data_stage1 import (
    Stage1MCDH5Dataset,
    SubjectBalancedBatchSampler,
)
from io_utils import load_json, save_json
from model_stage1 import Encoder, ProjectionHead, contrastive_loss_components


RUNTIME_JSON = OUTPUT_ROOT / "preflight" / "resolved_runtime.json"


def _probe_batch_size(recordings, batch_size: int, device: torch.device) -> Dict:
    ds = Stage1MCDH5Dataset(recordings, training=True)
    sampler = SubjectBalancedBatchSampler(recordings, batch_size=batch_size)
    sampler.set_epoch(0)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    encoder = Encoder().to(device)
    projector = ProjectionHead(
        PROJ_INPUT_DIM,
        PROJ_HIDDEN_DIM,
        PROJ_OUTPUT_DIM,
        PROJ_DROPOUT,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=1e-5,
    )

    batch = next(iter(loader))
    x1, x2, _meta = batch
    x1 = x1.to(device)
    x2 = x2.to(device)

    optimizer.zero_grad(set_to_none=True)
    comp = contrastive_loss_components(
        projector(encoder(x1)),
        projector(encoder(x2)),
        TEMPERATURE,
        MARGIN,
        TEMPORAL_WEIGHT,
    )
    comp["total"].backward()
    optimizer.step()

    peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    return {
        "batch_size": batch_size,
        "probe_loss": float(comp["total"].detach().cpu()),
        "peak_cuda_memory_bytes": peak,
    }


def resolve_or_load_runtime(recordings, device: torch.device, reprobe=False) -> Dict:
    if RUNTIME_JSON.exists() and not reprobe:
        payload = load_json(RUNTIME_JSON)
        if int(payload["batch_size"]) < 2:
            raise RuntimeError("Frozen runtime batch size is invalid.")
        return payload

    if device.type != "cuda":
        payload = {
            "device": str(device),
            "batch_size": DEFAULT_CPU_BATCH_SIZE,
            "selection": "fixed CPU fallback",
            "probe_results": [],
        }
        save_json(RUNTIME_JSON, payload)
        return payload

    probe_results = []
    chosen = None

    for batch_size in BATCH_SIZE_CANDIDATES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gc.collect()

        try:
            result = _probe_batch_size(recordings, batch_size, device)
            result["status"] = "PASS"
            probe_results.append(result)
            chosen = batch_size
            break
        except torch.cuda.OutOfMemoryError as exc:
            probe_results.append(
                {
                    "batch_size": batch_size,
                    "status": "CUDA_OOM",
                    "message": str(exc),
                }
            )
            torch.cuda.empty_cache()
            gc.collect()

    if chosen is None:
        raise RuntimeError(
            "None of the configured physical batch sizes fit on the GPU."
        )

    props = torch.cuda.get_device_properties(device)
    payload = {
        "device": str(device),
        "cuda_device_name": props.name,
        "cuda_total_memory_bytes": int(props.total_memory),
        "batch_size": int(chosen),
        "selection": "largest candidate passing a real forward+backward Stage-1 step",
        "probe_results": probe_results,
        "scientific_note": (
            "The resolved physical contrastive batch is frozen before training "
            "and must be reported with the experiment."
        ),
    }
    save_json(RUNTIME_JSON, payload)
    return payload
