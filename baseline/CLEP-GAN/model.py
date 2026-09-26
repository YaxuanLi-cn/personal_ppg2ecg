"""CLEP-GAN's paired 1-D adaptation for 10 s, 125 Hz MIMIC windows.

Implements the paper's Attention U-Net generators, symmetric NT-Xent feature
alignment, and ECG time/STFT adversaries. The official repository's data path
targets 128-sample BIDMC segments; this implementation keeps the architecture
and objective while adapting its tensor shapes to the repository's 1250 points.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.skip = nn.Conv1d(channels, hidden, 1, bias=False)
        self.gate = nn.Conv1d(channels, hidden, 1, bias=False)
        self.score = nn.Conv1d(hidden, 1, 1)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = F.interpolate(gate, size=skip.shape[-1], mode="linear", align_corners=False)
        alpha = torch.sigmoid(self.score(F.relu(self.skip(skip) + self.gate(gate))))
        return skip * alpha


class AttentionUNet1D(nn.Module):
    """Paper-style six-level attention U-Net, supporting encode/decode transfer."""
    def __init__(self, base_channels: int = 64, depth: int = 6, kernel_size: int = 31):
        super().__init__()
        if depth < 3 or kernel_size % 2 != 1:
            raise ValueError("depth must be >=3 and kernel_size must be odd")
        self.channels = [base_channels * min(2**i, 8) for i in range(depth)]
        self.down = nn.ModuleList()
        self.down_norm = nn.ModuleList()
        in_channels = 1
        for i, out_channels in enumerate(self.channels):
            self.down.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                       stride=2, padding=kernel_size // 2, bias=False))
            self.down_norm.append(nn.Identity() if i == 0 else nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        self.bottleneck = nn.Conv1d(self.channels[-1], self.channels[-1], 3, padding=1)
        self.up = nn.ModuleList()
        self.up_norm = nn.ModuleList()
        self.gates = nn.ModuleList()
        current = self.channels[-1]
        for skip_channels in reversed(self.channels[:-1]):
            self.up.append(nn.ConvTranspose1d(current, skip_channels, 4, stride=2, padding=1, bias=False))
            self.up_norm.append(nn.BatchNorm1d(skip_channels))
            self.gates.append(AttentionGate(skip_channels))
            current = skip_channels
        self.out = nn.Conv1d(current, 1, 7, padding=3)

        # Projection heads are the multimodal embedding maps used by NT-Xent.
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(self.channels[-2], 256), nn.LayerNorm(256)),
            nn.Sequential(nn.Linear(self.channels[-3], 256), nn.LayerNorm(256)),
        ])

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        h, features = x, []
        for conv, norm in zip(self.down, self.down_norm):
            h = F.leaky_relu(norm(conv(h)), 0.2, inplace=True)
            features.append(h)
        features[-1] = F.leaky_relu(self.bottleneck(features[-1]), 0.2, inplace=True)
        return features

    def decode(self, features: list[torch.Tensor]) -> torch.Tensor:
        h = features[-1]
        for i, (up, norm, gate) in enumerate(zip(self.up, self.up_norm, self.gates)):
            skip = features[-2-i]
            h = F.relu(norm(up(h)), inplace=True)
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = h + gate(skip, h)
        return torch.tanh(self.out(F.interpolate(h, size=features[0].shape[-1] * 2,
                                                 mode="linear", align_corners=False)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = self.encode(x)
        return self.decode(features), features

    def project_features(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        selected = (features[-2], features[-3])
        return [F.normalize(head(value.mean(dim=-1)), dim=-1)
                for head, value in zip(self.projections, selected)]


class TimeDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for i in range(5):
            out_channels = min(base_channels * 2**i, base_channels * 8)
            layers.append(nn.Conv1d(in_channels, out_channels, 16, stride=2, padding=7))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
        layers.append(nn.Conv1d(in_channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectrogramDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 32, n_fft: int = 254, hop_length: int = 4):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        layers: list[nn.Module] = []
        in_channels = 1
        for i in range(4):
            out_channels = base_channels * 2**i
            layers.extend([nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                           nn.LeakyReLU(0.2, inplace=True)])
            if i:
                layers.insert(len(layers)-1, nn.InstanceNorm2d(out_channels, affine=True))
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        spec = torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.n_fft, window=window, return_complex=True).abs()
        return self.net(torch.log(spec.clamp_min(1e-10)).unsqueeze(1))


class CLEPGAN(nn.Module):
    def __init__(self, base_channels: int = 64, depth: int = 6, kernel_size: int = 31,
                 discriminator_channels: int = 64):
        super().__init__()
        self.ppg_generator = AttentionUNet1D(base_channels, depth, kernel_size)
        self.ecg_generator = AttentionUNet1D(base_channels, depth, kernel_size)
        self.ecg_time_discriminator = TimeDiscriminator(discriminator_channels)
        self.ecg_freq_discriminator = SpectrogramDiscriminator(max(16, discriminator_channels // 2))
        self.logit_scale = nn.Parameter(torch.tensor(0.07).log())

    def contrastive(self, ppg_features: list[torch.Tensor], ecg_features: list[torch.Tensor]):
        return symmetric_nt_xent(self.ppg_generator.project_features(ppg_features),
                                 self.ecg_generator.project_features(ecg_features),
                                 self.logit_scale)

    def forward(self, ppg: torch.Tensor, ecg: torch.Tensor | None = None):
        ppg_recon, ppg_features = self.ppg_generator(ppg)
        ecg_recon, ecg_features = self.ecg_generator(ecg) if ecg is not None else (None, None)
        ppg_to_ecg = self.ecg_generator.decode(ppg_features)
        return ppg_recon, ecg_recon, ppg_to_ecg, ppg_features, ecg_features


def symmetric_nt_xent(ppg_features: list[torch.Tensor], ecg_features: list[torch.Tensor],
                      logit_scale: torch.Tensor) -> torch.Tensor:
    """Cross-modal in-batch retrieval loss, averaged over two encoder levels."""
    scale = logit_scale.exp().clamp(max=100)
    losses = []
    for ppg_z, ecg_z in zip(ppg_features, ecg_features):
        logits = scale * (ppg_z @ ecg_z.T)
        labels = torch.arange(logits.shape[0], device=logits.device)
        losses.append((F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5)
    return torch.stack(losses).mean()
