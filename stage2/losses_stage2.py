from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_stage2 import (
    AUX_CORR_WEIGHT,
    AUX_WAVE_WEIGHT,
    AUX_WEIGHT,
    MORPH_MULTISCALE_LAGS,
    MORPH_NORM_EPS,
    PSD_NFFT,
    PULSE_BAND_HZ,
    SMOOTH_L1_BETA,
    W_BAND,
    W_CORR,
    W_DERIVATIVE,
    W_MORPH_D1,
    W_MORPH_D2,
    W_MORPH_MULTISCALE,
    W_SPECTRUM,
    W_STD,
    W_WAVE,
)


def pearson_per_sample(pred, target, eps=1e-8):
    p = pred - pred.mean(dim=1, keepdim=True)
    y = target - target.mean(dim=1, keepdim=True)
    num = (p * y).sum(dim=1)
    den = torch.sqrt((p.square().sum(dim=1) + eps) * (y.square().sum(dim=1) + eps))
    return num / den


def smooth_l1_per_sample(pred, target):
    return F.smooth_l1_loss(
        pred, target, beta=SMOOTH_L1_BETA, reduction="none"
    ).mean(dim=1)


def derivative_per_sample(pred, target):
    # ORIGINAL v3 first-derivative term, preserved unchanged.
    return smooth_l1_per_sample(
        pred[:, 1:] - pred[:, :-1],
        target[:, 1:] - target[:, :-1],
    )


def _zshape(x, eps=MORPH_NORM_EPS):
    """Per-sample normalization used only inside the added morphology losses."""
    x = x - x.mean(dim=1, keepdim=True)
    s = x.std(dim=1, keepdim=True, unbiased=False)
    return x / (s + eps)


def normalized_d1_shape_per_sample(pred, target):
    """Preserve the normalized local slope pattern independent of global scale."""
    dp = pred[:, 1:] - pred[:, :-1]
    dy = target[:, 1:] - target[:, :-1]
    return smooth_l1_per_sample(_zshape(dp), _zshape(dy))


def normalized_d2_shape_per_sample(pred, target):
    """
    Preserve local curvature/inflection structure.
    This directly targets the fidelity loss observed in the OOF morphology audit.
    """
    dp = pred[:, 1:] - pred[:, :-1]
    dy = target[:, 1:] - target[:, :-1]
    ddp = dp[:, 1:] - dp[:, :-1]
    ddy = dy[:, 1:] - dy[:, :-1]
    return smooth_l1_per_sample(_zshape(ddp), _zshape(ddy))


def multiscale_shape_per_sample(pred, target):
    """
    Compare normalized temporal increments at lags 1, 2 and 4.
    This discourages oversmoothing while remaining differentiable and
    independent of explicit peak/notch detection.
    """
    terms = []
    for lag in MORPH_MULTISCALE_LAGS:
        if pred.shape[1] <= lag:
            continue
        dp = pred[:, lag:] - pred[:, :-lag]
        dy = target[:, lag:] - target[:, :-lag]
        terms.append(smooth_l1_per_sample(_zshape(dp), _zshape(dy)))
    if not terms:
        return pred.new_zeros((pred.shape[0],))
    return torch.stack(terms, dim=1).mean(dim=1)


def _spectral_components(pred, target, fps):
    # ORIGINAL v3 spectral term, preserved unchanged.
    p = pred - pred.mean(dim=1, keepdim=True)
    y = target - target.mean(dim=1, keepdim=True)
    P = torch.fft.rfft(p, n=PSD_NFFT, dim=1).abs().square()
    Y = torch.fft.rfft(y, n=PSD_NFFT, dim=1).abs().square()

    B, F = P.shape
    k = torch.arange(F, device=P.device, dtype=P.dtype).unsqueeze(0)
    fps = fps.to(device=P.device, dtype=P.dtype).reshape(-1, 1)
    hz = k * fps / float(PSD_NFFT)
    mask = (hz >= PULSE_BAND_HZ[0]) & (hz <= PULSE_BAND_HZ[1])
    mask_f = mask.to(P.dtype)

    P_band = P * mask_f
    Y_band = Y * mask_f

    P_norm = P_band / (P_band.sum(dim=1, keepdim=True) + 1e-8)
    Y_norm = Y_band / (Y_band.sum(dim=1, keepdim=True) + 1e-8)

    spectrum = (P_norm - Y_norm).abs().sum(dim=1) * 0.5

    total_pred = P[:, 1:].sum(dim=1) + 1e-8
    band_pred = P_band.sum(dim=1)
    band = 1.0 - torch.clamp(band_pred / total_pred, 0.0, 1.0)
    return spectrum, band


class RPPGPhysioCompositeLoss(nn.Module):
    """
    Stage-2 v4 morphology-preserving objective.

    ORIGINAL v3 objective (unchanged):
      3.00 * (1 - Pearson)
    + 0.75 * SmoothL1 waveform
    + 0.50 * SmoothL1 raw first derivative
    + 0.75 * normalized pulse-band spectral distance
    + 0.05 * out-of-band energy penalty
    + 0.25 * |std(pred)-std(target)|
    + 0.15 * mean(aux_corr + aux_wave)

    ONLY v4 additions:
    + 0.15 * normalized first-derivative shape loss
    + 0.35 * normalized second-derivative shape loss
    + 0.20 * normalized multi-scale temporal-increment loss (lags 1,2,4)

    No BP, rigidity, HR, HRV, RR, attention or target-derived handcrafted
    peak/notch detector is used.
    """

    def forward(self, out, target, fps):
        pred = out["rppg"]

        # ---------------- ORIGINAL v3 components ----------------
        corr = pearson_per_sample(pred, target)
        corr_loss = 1.0 - corr
        wave = smooth_l1_per_sample(pred, target)
        derivative = derivative_per_sample(pred, target)
        spectrum, band = _spectral_components(pred, target, fps)

        pred_std = pred.std(dim=1, unbiased=False)
        target_std = target.std(dim=1, unbiased=False)
        std_loss = (pred_std - target_std).abs()

        baseline_final_per = (
            W_CORR * corr_loss
            + W_WAVE * wave
            + W_DERIVATIVE * derivative
            + W_SPECTRUM * spectrum
            + W_BAND * band
            + W_STD * std_loss
        )

        # ---------------- ONLY v4 additions ----------------
        morph_d1 = normalized_d1_shape_per_sample(pred, target)
        morph_d2 = normalized_d2_shape_per_sample(pred, target)
        morph_multiscale = multiscale_shape_per_sample(pred, target)

        morphology_per = (
            W_MORPH_D1 * morph_d1
            + W_MORPH_D2 * morph_d2
            + W_MORPH_MULTISCALE * morph_multiscale
        )

        final_per = baseline_final_per + morphology_per

        # ---------------- ORIGINAL v3 auxiliary supervision ----------------
        aux_losses = []
        for aux in out["aux_rppg"]:
            aux_corr = 1.0 - pearson_per_sample(aux, target)
            aux_wave = smooth_l1_per_sample(aux, target)
            aux_losses.append(
                AUX_CORR_WEIGHT * aux_corr + AUX_WAVE_WEIGHT * aux_wave
            )

        if aux_losses:
            aux_stack = torch.stack(aux_losses, dim=1)
            aux_mean = aux_stack.mean(dim=1)
        else:
            aux_stack = final_per.new_zeros((len(final_per), 0))
            aux_mean = final_per.new_zeros(final_per.shape)

        total_per = final_per + AUX_WEIGHT * aux_mean

        parts = {
            "total": total_per.mean(),
            "total_per_sample": total_per,
            "final": final_per.mean(),
            "final_per_sample": final_per,
            "baseline_final": baseline_final_per.mean(),
            "baseline_final_per_sample": baseline_final_per,
            "morphology": morphology_per.mean(),
            "morphology_per_sample": morphology_per,
            "morph_d1": morph_d1.mean(),
            "morph_d1_per_sample": morph_d1,
            "morph_d2": morph_d2.mean(),
            "morph_d2_per_sample": morph_d2,
            "morph_multiscale": morph_multiscale.mean(),
            "morph_multiscale_per_sample": morph_multiscale,
            "corr_loss": corr_loss.mean(),
            "corr_loss_per_sample": corr_loss,
            "wave": wave.mean(),
            "wave_per_sample": wave,
            "derivative": derivative.mean(),
            "derivative_per_sample": derivative,
            "spectrum": spectrum.mean(),
            "spectrum_per_sample": spectrum,
            "band": band.mean(),
            "band_per_sample": band,
            "std_loss": std_loss.mean(),
            "std_loss_per_sample": std_loss,
            "aux_mean": aux_mean.mean(),
            "aux_mean_per_sample": aux_mean,
            "aux_per_sample": aux_stack,
        }
        return parts["total"], parts
