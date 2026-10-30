from __future__ import annotations
import gc
import torch
from torch.utils.data import DataLoader
from config_stage3 import *
from data_stage3 import subject_state_collate
from model_stage3 import PulseBridgeStage3DualPathV7
from losses_stage3 import DualPathLossV11
from paired_loss_v19 import paired_coupled_d03_loss_v19
from io_utils import save_json, load_json

RUNTIME_JSON = OUTPUT_ROOT / 'preflight/resolved_runtime.json'
RUNTIME_VERSION = V11_VERSION


def probe(dataset, device, batch_size):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=subject_state_collate)
    waves, prez_waves, latents, amp_features, phys_features, mask, cams, yn, _, metas = next(iter(dl))
    waves = waves.to(device)
    prez_waves = prez_waves.to(device)
    latents = latents.to(device)
    amp_features = amp_features.to(device)
    phys_features = phys_features.to(device)
    mask = mask.to(device)
    cams = cams.to(device)
    yn = yn.to(device)
    m = PulseBridgeStage3DualPathV7().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = DualPathLossV11()
    opt.zero_grad(set_to_none=True)
    out=m(waves, prez_waves, mask, cams, latents, amp_features, phys_features)
    out=m.apply_v19_coupled_sbp_correction(out,metas)
    loss, _ = crit(out, yn)
    # Include the new paired branch in the CUDA memory probe. The probe batch is
    # not guaranteed to contain a complete pair, which is safe: the paired loss
    # becomes zero in that case.
    pair_loss, _, _, _ = paired_coupled_d03_loss_v19(m,out,yn,metas)
    loss = loss + V14_PAIR_WEIGHT*pair_loss
    loss.backward()
    opt.step()
    return float(loss.detach().cpu())


def resolve_runtime(dataset, device):
    if RUNTIME_JSON.exists():
        x = load_json(RUNTIME_JSON)
        if x.get('runtime_version') == RUNTIME_VERSION:
            return x
    if device.type != 'cuda':
        x = {'runtime_version':RUNTIME_VERSION,'device':str(device),'batch_size':DEFAULT_CPU_BATCH_SIZE}
        save_json(RUNTIME_JSON, x)
        return x
    rows = []
    chosen = None
    for b in BATCH_SIZE_CANDIDATES:
        torch.cuda.empty_cache(); gc.collect()
        try:
            loss = probe(dataset, device, b)
            rows.append({'batch':b,'status':'PASS','loss':loss})
            chosen = b
            break
        except torch.cuda.OutOfMemoryError:
            rows.append({'batch':b,'status':'CUDA_OOM'})
            torch.cuda.empty_cache()
    if chosen is None:
        raise RuntimeError('No Stage3-v19-CoupledD03 batch fits GPU')
    x = {'runtime_version':RUNTIME_VERSION,'device':str(device),'batch_size':chosen,'probe':rows}
    save_json(RUNTIME_JSON, x)
    return x
