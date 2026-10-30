from __future__ import annotations
import torch
from config_stage3 import *


def _role(state):
    x = str(state).strip().lower()
    if 'before' in x:
        return 0
    if 'after' in x:
        return 1
    return None


def _ccc_loss(p, t, eps=PAIR_DELTA_EPS):
    pm = p.mean()
    tm = t.mean()
    pv = ((p - pm) ** 2).mean()
    tv = ((t - tm) ** 2).mean()
    cov = ((p - pm) * (t - tm)).mean()
    return 1.0 - 2.0 * cov / (pv + tv + (pm - tm) ** 2 + eps)


def paired_delta_loss_v11(out, yn, metas):
    """Structured Before->After objective in normalized target space.

    The MAE term preserves delta magnitude.  The CCC term is activated only
    when enough complete subject pairs exist in the batch, avoiding unstable
    correlation estimates for tiny batches.
    """
    pred = torch.stack([out['sbp'], out['dbp'], out['stiffness']], 1)
    by_sid = {}
    for i, m in enumerate(metas):
        by_sid.setdefault(str(m['subject_id']), []).append(i)

    dp_list = []
    dt_list = []
    for _, idxs in by_sid.items():
        if len(idxs) != 2:
            continue
        a, b = idxs
        ra, rb = _role(metas[a]['state']), _role(metas[b]['state'])
        if ra is not None and rb is not None and ra != rb:
            ib = a if ra == 0 else b
            ia = a if ra == 1 else b
        else:
            pair = sorted(idxs, key=lambda j: str(metas[j]['state']).lower())
            ib, ia = pair[0], pair[1]
        dp_list.append(pred[ia] - pred[ib])
        dt_list.append(yn[ia] - yn[ib])

    if not dp_list:
        zero = pred.sum() * 0.0
        return zero, {
            'paired_delta_mae': zero,
            'paired_delta_ccc_loss': zero,
            'paired_sbp_delta_mae': zero,
            'paired_dbp_delta_mae': zero,
            'paired_stiffness_delta_mae': zero,
        }, 0

    dp = torch.stack(dp_list, 0)
    dt = torch.stack(dt_list, 0)
    mae_t = torch.abs(dp - dt).mean(0)
    weighted_mae = (
        V11_SBP_WEIGHT * mae_t[0]
        + V11_DBP_WEIGHT * mae_t[1]
        + V11_STIFFNESS_WEIGHT * mae_t[2]
    )

    if dp.shape[0] >= PAIR_DELTA_MIN_PAIRS_FOR_CCC:
        ccc_t = torch.stack([_ccc_loss(dp[:, j], dt[:, j]) for j in range(3)])
        weighted_ccc = (
            V11_SBP_WEIGHT * ccc_t[0]
            + V11_DBP_WEIGHT * ccc_t[1]
            + V11_STIFFNESS_WEIGHT * ccc_t[2]
        )
    else:
        weighted_ccc = weighted_mae * 0.0

    total = PAIR_DELTA_MAE_WEIGHT * weighted_mae + PAIR_DELTA_CCC_WEIGHT * weighted_ccc
    return total, {
        'paired_delta_mae': weighted_mae,
        'paired_delta_ccc_loss': weighted_ccc,
        'paired_sbp_delta_mae': mae_t[0],
        'paired_dbp_delta_mae': mae_t[1],
        'paired_stiffness_delta_mae': mae_t[2],
    }, int(dp.shape[0])
