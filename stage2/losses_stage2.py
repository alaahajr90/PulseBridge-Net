from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_stage2 import (
    AUX_CORR_WEIGHT,
    AUX_WAVE_WEIGHT,
    AUX_WEIGHT,
    PSD_NFFT,
    PULSE_BAND_HZ,
    SMOOTH_L1_BETA,
    W_BAND,
    W_CORR,
    W_DERIVATIVE,
    W_SPECTRUM,
    W_STD,
    W_WAVE,
    W_MORPH_FIRST,
    W_MORPH_SECOND,
    W_MORPH_MULTISCALE,
    MORPH_LAGS,
    MORPH_Z_EPS,
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
    return smooth_l1_per_sample(
        pred[:, 1:] - pred[:, :-1],
        target[:, 1:] - target[:, :-1],
    )


def _spectral_components(pred, target, fps):
    # Demean; rFFT power.
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

    total_pred = P[:, 1:].sum(dim=1) + 1e-8  # exclude DC
    band_pred = P_band.sum(dim=1)
    band = 1.0 - torch.clamp(band_pred / total_pred, 0.0, 1.0)
    return spectrum, band


def z_normalize(x: torch.Tensor, eps: float = MORPH_Z_EPS) -> torch.Tensor:
    """Eq. (11): standardize each sample using population std (unbiased=False)."""
    return (x - x.mean(dim=1, keepdim=True)) / (
        x.std(dim=1, keepdim=True, unbiased=False) + eps
    )


def first_difference(x: torch.Tensor, lag: int = 1) -> torch.Tensor:
    """Temporal increment delta_lag x[t] = x[t + lag] - x[t]."""
    if x.ndim != 2 or lag < 1 or x.size(1) <= lag:
        raise ValueError(f"Expected [B,T] with T>lag={lag}; got {tuple(x.shape)}")
    return x[:, lag:] - x[:, :-lag]


def second_difference(x: torch.Tensor) -> torch.Tensor:
    """Second temporal difference, delta(delta x), NOT the lag-2 increment."""
    return first_difference(first_difference(x))


def morphology_terms_per_sample(pred: torch.Tensor, target: torch.Tensor):
    """Manuscript Eqs. (12)–(14): normalized derivatives and lagged increments."""
    if pred.shape != target.shape or pred.ndim != 2 or pred.size(1) <= max(MORPH_LAGS):
        raise ValueError("Morphology requires two matching [B,T] waveforms, T>max(lags)")

    morph_first = smooth_l1_per_sample(
        z_normalize(first_difference(pred)),
        z_normalize(first_difference(target)),
    )
    morph_second = smooth_l1_per_sample(
        z_normalize(second_difference(pred)),
        z_normalize(second_difference(target)),
    )
    morph_multiscale = torch.stack([
        smooth_l1_per_sample(
            z_normalize(first_difference(pred, lag)),
            z_normalize(first_difference(target, lag)),
        )
        for lag in MORPH_LAGS
    ], dim=1).mean(dim=1)
    return morph_first, morph_second, morph_multiscale


class RPPGPhysioCompositeLoss(nn.Module):
    """Manuscript-aligned Stage-2 composite objective, Eqs. (15)–(17).

    Derived from the supplied v3 archive. Three morphology terms were added
    for source-level alignment; no claim is made that this file generated the
    previously reported experimental checkpoints or results.
    """

    def forward(self, out, target, fps):
        pred = out["rppg"]

        corr = pearson_per_sample(pred, target)
        corr_loss = 1.0 - corr
        wave = smooth_l1_per_sample(pred, target)
        derivative = derivative_per_sample(pred, target)
        spectrum, band = _spectral_components(pred, target, fps)

        pred_std = pred.std(dim=1, unbiased=False)
        target_std = target.std(dim=1, unbiased=False)
        std_loss = (pred_std - target_std).abs()
        morph_first, morph_second, morph_multiscale = morphology_terms_per_sample(pred, target)

        final_per = (
            W_CORR * corr_loss
            + W_WAVE * wave
            + W_DERIVATIVE * derivative
            + W_SPECTRUM * spectrum
            + W_BAND * band
            + W_STD * std_loss
            + W_MORPH_FIRST * morph_first
            + W_MORPH_SECOND * morph_second
            + W_MORPH_MULTISCALE * morph_multiscale
        )

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
            "morph_first": morph_first.mean(),
            "morph_first_per_sample": morph_first,
            "morph_second": morph_second.mean(),
            "morph_second_per_sample": morph_second,
            "morph_multiscale": morph_multiscale.mean(),
            "morph_multiscale_per_sample": morph_multiscale,
            "aux_mean": aux_mean.mean(),
            "aux_mean_per_sample": aux_mean,
            "aux_per_sample": aux_stack,
        }
        return parts["total"], parts
