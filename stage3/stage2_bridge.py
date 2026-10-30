from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import torch
from config_stage3 import STAGE2_DIR, STAGE2_MODEL_PY

def _load_stage2_module():
    # Temporarily add Stage2 only while executing its own module because
    # model_stage2 imports config_stage2/io_utils by bare name. Remove and
    # restore conflicting modules immediately afterward to avoid shadowing Stage3.
    stage2 = str(STAGE2_DIR)
    saved = {}
    for name in ("config_stage2","io_utils","model_stage2"):
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, stage2)
        spec = importlib.util.spec_from_file_location(
            "pulsebridge_stage2_model_for_stage3", STAGE2_MODEL_PY
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {STAGE2_MODEL_PY}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = old_path
        for name in ("config_stage2","io_utils","model_stage2"):
            sys.modules.pop(name, None)
        sys.modules.update(saved)

_STAGE2 = _load_stage2_module()
Stage2RPPGModel = _STAGE2.Stage2RPPGModel

def load_stage2_model(checkpoint, device):
    checkpoint=Path(checkpoint)
    if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    state=ck["model"] if isinstance(ck,dict) and "model" in ck else ck
    model=Stage2RPPGModel()
    missing,unexpected=model.load_state_dict(state,strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Stage-2 model mismatch missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    for p in model.parameters(): p.requires_grad=False
    return model
