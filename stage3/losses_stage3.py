from __future__ import annotations
import torch
import torch.nn as nn
from config_stage3 import *


def ccc_loss(p, t, eps=1e-8):
    pm = p.mean()
    tm = t.mean()
    pv = ((p - pm) ** 2).mean()
    tv = ((t - tm) ** 2).mean()
    cov = ((p - pm) * (t - tm)).mean()
    return 1.0 - 2.0 * cov / (pv + tv + (pm - tm) ** 2 + eps)


def log_std_match_loss(p, t, eps=V11_VARIANCE_EPS):
    """Penalize variance compression/expansion without using raw target units."""
    ps = torch.sqrt(((p - p.mean()) ** 2).mean() + eps)
    ts = torch.sqrt(((t - t.mean()) ** 2).mean() + eps)
    return torch.abs(torch.log(ps + eps) - torch.log(ts + eps))


class DualPathLossV11(nn.Module):
    """v10 absolute loss + explicit variance-preservation regularizer."""

    def one(self, p, t):
        e = p - t
        mae = e.abs().mean()
        rmse = torch.sqrt(e.square().mean() + 1e-8)
        ccc = ccc_loss(p, t)
        var = log_std_match_loss(p, t)
        base = V11_MAE_WEIGHT * mae + V11_RMSE_WEIGHT * rmse + V11_CCC_WEIGHT * ccc
        total = base + V11_VARIANCE_WEIGHT * var
        return total, mae, rmse, ccc, var

    def forward(self, o, y):
        sl, smae, srmse, sccc, svar = self.one(o['sbp'], y[:, 0])
        dl, dmae, drmse, dccc, dvar = self.one(o['dbp'], y[:, 1])
        rl, rmae, rrmse, rccc, rvar = self.one(o['stiffness'], y[:, 2])

        reg = V11_SBP_WEIGHT * sl + V11_DBP_WEIGHT * dl + V11_STIFFNESS_WEIGHT * rl
        bp = V11_SBP_WEIGHT * sl + V11_DBP_WEIGHT * dl

        return reg, {
            'reg_main': reg,
            'bp_main': bp,
            'stiffness_main': rl,
            'sbp_direct_mae': smae,
            'sbp_direct_rmse': srmse,
            'dbp_direct_mae': dmae,
            'dbp_direct_rmse': drmse,
            'stiffness_direct_mae': rmae,
            'stiffness_direct_rmse': rrmse,
            'sbp_ccc_loss': sccc,
            'dbp_ccc_loss': dccc,
            'stiffness_ccc_loss': rccc,
            'sbp_variance_match_loss': svar,
            'dbp_variance_match_loss': dvar,
            'stiffness_variance_match_loss': rvar,
        }
