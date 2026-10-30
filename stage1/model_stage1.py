import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Original PulseBridge-Net Stage-1 3D convolutional encoder.

    The final four actual Conv3d layers remain:
      convblock2.0, convblock2.3, convblock3.0, convblock3.3
    which is important for Stage-2 partial fine-tuning.
    """

    def __init__(self, in_channels=3):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, (1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool3d((1, 2, 2)),
        )
        self.convblock1 = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool3d((1, 2, 2)),
        )
        self.convblock2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool3d((1, 2, 2)),
        )
        self.convblock3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

    @staticmethod
    def _to_seq(x):
        return (
            F.adaptive_avg_pool3d(x, (x.shape[2], 1, 1))
            .squeeze(-1)
            .squeeze(-1)
            .permute(0, 2, 1)
        )

    def extract_features(self, x):
        p5 = self.conv1(x)
        p6 = self.convblock1(p5)
        p7 = self.convblock2(p6)
        p = self.convblock3(p7)
        return [self._to_seq(y) for y in (p5, p6, p7, p)]

    def forward(self, x):
        p5 = self.conv1(x)
        p6 = self.convblock1(p5)
        p7 = self.convblock2(p6)
        p = self.convblock3(p7)
        return self.global_pool(p).squeeze(-1).squeeze(-1).permute(0, 2, 1)


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim=64,
        hidden_dim=64,
        output_dim=64,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        b, t, c = x.shape
        return self.mlp(x.reshape(b * t, c)).view(b, t, -1)


def temporal_consistency_loss(z, delta=1):
    if z.shape[1] <= delta:
        return z.new_tensor(0.0)
    return (
        1
        - F.cosine_similarity(
            z[:, :-delta, :],
            z[:, delta:, :],
            dim=-1,
        )
    ).mean()


def contrastive_loss_components(
    z1,
    z2,
    temperature=0.07,
    margin=0.30,
    temporal_weight=0.25,
):
    if z1.shape != z2.shape or z1.ndim != 3:
        raise ValueError(f"Expected matching [B,T,D] tensors, got {z1.shape}, {z2.shape}")

    batch, time, dim = z1.shape
    if batch < 2:
        raise ValueError("Contrastive loss requires at least 2 recordings per batch.")

    n = batch * time
    a = F.normalize(z1.reshape(n, dim), dim=1)
    b = F.normalize(z2.reshape(n, dim), dim=1)
    z = torch.cat([a, b], dim=0)

    sim = z @ z.T / temperature
    fill = -1e4 if sim.dtype == torch.float16 else -1e9
    eye = torch.eye(2 * n, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, fill)

    positive_logits = (a * b).sum(dim=1) / temperature
    denominator = torch.exp(sim - margin).sum(dim=1)[:n]
    contrastive = -torch.log(
        torch.exp(positive_logits) / (denominator + 1e-12)
    ).mean()

    temporal = temporal_weight * (
        temporal_consistency_loss(z1)
        + temporal_consistency_loss(z2)
    )
    total = contrastive + temporal

    positive_cosine = (a * b).sum(dim=1).mean()
    cosine = z @ z.T
    nonself = ~eye
    negative_cosine = cosine[nonself].mean()
    embedding_std = z.std(dim=0).mean()

    return {
        "total": total,
        "contrastive": contrastive,
        "temporal": temporal,
        "positive_cosine": positive_cosine,
        "negative_cosine": negative_cosine,
        "cosine_gap": positive_cosine - negative_cosine,
        "embedding_std": embedding_std,
    }
