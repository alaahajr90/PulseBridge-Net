from __future__ import annotations

import importlib.util
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from config_stage2 import (
    DECODER_CHANNELS,
    DECODER_DILATIONS,
    DECODER_DROPOUT,
    DECODER_KERNEL_SIZE,
    FREEZE_ALL_ENCODER_BATCHNORM_AFFINE,
    FREEZE_ALL_ENCODER_BATCHNORM_STATS,
    LATE_CONV_COUNT,
    STAGE1_MODEL_PY,
)
from io_utils import torch_load_weights


def _load_stage1_encoder_class():
    path = Path(STAGE1_MODEL_PY)
    if not path.exists():
        raise FileNotFoundError(f"Stage-1 model source missing: {path}")
    spec = importlib.util.spec_from_file_location(
        "pulsebridge_stage1_model_for_stage2_v3", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Stage-1 model: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Encoder"):
        raise ImportError(f"Stage-1 model has no Encoder: {path}")
    return module.Encoder


Encoder = _load_stage1_encoder_class()


class WaveformProjectionHead(nn.Module):
    """Light auxiliary head; final prediction does NOT use this class."""
    def __init__(self, in_channels=64):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, seq):
        return self.proj(seq.transpose(1, 2)).squeeze(1)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, dilation, kernel_size=5, dropout=0.1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad, dilation=dilation, bias=False
        )
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad, dilation=dilation, bias=False
        )
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class TemporalRPPGDecoder(nn.Module):
    """
    Attention-free residual TCN. Receptive field spans multiple pulse samples,
    unlike the v2 1x1 projection that encouraged a near-zero solution.
    """
    def __init__(self, in_channels=64):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, DECODER_CHANNELS, 3, padding=1)
        self.blocks = nn.ModuleList([
            TemporalResidualBlock(
                DECODER_CHANNELS,
                dilation=d,
                kernel_size=DECODER_KERNEL_SIZE,
                dropout=DECODER_DROPOUT,
            )
            for d in DECODER_DILATIONS
        ])
        self.out = nn.Sequential(
            nn.Conv1d(DECODER_CHANNELS, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 1, 1),
        )

    def forward(self, seq):
        x = seq.transpose(1, 2)
        x = F.gelu(self.in_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.out(x).squeeze(1)


class Stage2RPPGModel(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else Encoder()
        self.aux_heads = nn.ModuleList(
            [WaveformProjectionHead(64) for _ in range(3)]
        )
        self.decoder = TemporalRPPGDecoder(64)
        self._configure_finetuning()

    def _all_conv3d(self):
        return [
            (name, module)
            for name, module in self.encoder.named_modules()
            if isinstance(module, nn.Conv3d)
        ]

    def _configure_finetuning(self):
        # Freeze everything first.
        for p in self.encoder.parameters():
            p.requires_grad = False

        convs = self._all_conv3d()
        if len(convs) < LATE_CONV_COUNT:
            raise RuntimeError(
                f"Stage-1 encoder has only {len(convs)} Conv3d layers."
            )

        self.all_conv_names = [n for n, _ in convs]
        self.early_conv_names = [n for n, _ in convs[:-LATE_CONV_COUNT]]
        self.late_conv_names = [n for n, _ in convs[-LATE_CONV_COUNT:]]

        # All Conv3d parameters may adapt; optimizer applies discriminative LR.
        for _name, module in convs:
            for p in module.parameters(recurse=False):
                p.requires_grad = True

        # BN stays frozen, including affine parameters.
        if FREEZE_ALL_ENCODER_BATCHNORM_AFFINE:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm3d):
                    for p in module.parameters(recurse=False):
                        p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode and FREEZE_ALL_ENCODER_BATCHNORM_STATS:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm3d):
                    module.eval()
        return self

    @staticmethod
    def _seq(x):
        x = F.adaptive_avg_pool3d(x, (x.shape[2], 1, 1))
        return x.squeeze(-1).squeeze(-1).permute(0, 2, 1)

    def forward_late_features(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.convblock1(x)

        c4 = self.encoder.convblock2[0](x)
        c4 = self.encoder.convblock2[1](c4)
        c4 = self.encoder.convblock2[2](c4)

        c5 = self.encoder.convblock2[3](c4)
        c5 = self.encoder.convblock2[4](c5)
        c5 = self.encoder.convblock2[5](c5)

        x2 = self.encoder.convblock2[6](c5)

        c6 = self.encoder.convblock3[0](x2)
        c6 = self.encoder.convblock3[1](c6)
        c6 = self.encoder.convblock3[2](c6)

        c7 = self.encoder.convblock3[3](c6)
        c7 = self.encoder.convblock3[4](c7)
        c7 = self.encoder.convblock3[5](c7)

        return [self._seq(c4), self._seq(c5), self._seq(c6), self._seq(c7)]

    def forward(self, x):
        f4, f5, f6, f7 = self.forward_late_features(x)
        aux = [
            head(feat)
            for head, feat in zip(self.aux_heads, (f4, f5, f6))
        ]
        final = self.decoder(f7)
        return {"aux_rppg": aux, "rppg": final, "all_rppg": aux + [final]}

    def parameter_groups(self):
        by_name = dict(self.encoder.named_parameters())
        early_prefixes = tuple(n + "." for n in self.early_conv_names)
        late_prefixes = tuple(n + "." for n in self.late_conv_names)

        early = [
            p for n, p in by_name.items()
            if p.requires_grad and n.startswith(early_prefixes)
        ]
        late = [
            p for n, p in by_name.items()
            if p.requires_grad and n.startswith(late_prefixes)
        ]
        heads = list(self.aux_heads.parameters()) + list(self.decoder.parameters())
        return early, late, heads

    def trainable_parameter_report(self):
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        return {
            "all_conv_names": self.all_conv_names,
            "early_conv_names": self.early_conv_names,
            "late_conv_names": self.late_conv_names,
            "trainable": trainable,
            "n_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "n_total": sum(p.numel() for p in self.parameters()),
        }


def load_stage1_encoder_weights(path: Path, device="cpu"):
    encoder = Encoder()
    state = torch_load_weights(path, map_location=device)
    if isinstance(state, dict) and "encoder" in state:
        state = state["encoder"]
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Stage-1 encoder mismatch. missing={missing}, unexpected={unexpected}"
        )
    return encoder
