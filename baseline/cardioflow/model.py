"""Peak-aware conditional Rectified Flow used by CardioFlow.

CardioFlow keeps the waveform-space straight flow but makes the vector field
aware of fiducial structure.  The PPG peak mask is available at inference and
is supplied as a third condition channel.  During training, the velocity
regression is reweighted around ECG peaks (with a small PPG-mask term), which
prevents the high-energy QRS regions from being washed out by the long flat
parts of a window.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def peak_mask(signal: Tensor, *, width: int = 15, threshold: float = 0.35,
              dilation: int = 9) -> Tensor:
    """Return a differentiability-free soft binary mask around local peaks.

    ``signal`` is ``[B,L,1]`` or ``[B,L]``.  ECG uses absolute amplitude so
    inverted lead-II recordings are handled as well.  The operation is only a
    condition/weight and is detached from the flow network.
    """
    if signal.ndim == 3:
        if signal.shape[-1] != 1:
            raise ValueError("signal must have one channel")
        x = signal[..., 0]
    elif signal.ndim == 2:
        x = signal
    else:
        raise ValueError("signal must have shape [B,L,1] or [B,L]")
    if width < 3 or width % 2 == 0 or dilation < 1 or dilation % 2 == 0:
        raise ValueError("width must be odd >=3 and dilation must be odd")
    x = x.detach()
    x = x.abs()
    # Per-window robust scaling makes the mask usable for normalized and raw
    # signals without baking data statistics into the model.
    x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)
    local_max = F.max_pool1d(x[:, None], width, stride=1, padding=width // 2)[:, 0]
    candidates = (x >= local_max - 1e-6) & (x > threshold)
    mask = candidates.float()[:, None]
    if dilation > 1:
        mask = F.max_pool1d(mask, dilation, stride=1, padding=dilation // 2)
    return mask[:, 0, :, None]


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        time = time.float().reshape(-1)
        half = self.dim // 2
        if half == 0:
            return time[:, None]
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=time.device, dtype=time.dtype)
                         / max(half - 1, 1))
        emb = torch.cat((torch.sin(time[:, None] * freq[None]), torch.cos(time[:, None] * freq[None])), dim=-1)
        return F.pad(emb, (0, self.dim - emb.shape[-1]))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, 7, padding=3)
        self.time = nn.Linear(time_dim, 2 * channels)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, 7, padding=3)

    def forward(self, x: Tensor, time: Tensor) -> Tensor:
        h = self.norm1(x)
        scale, shift = self.time(time).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        return x + self.conv2(F.silu(self.norm2(self.conv1(F.silu(h)))))


class CardioFlow(nn.Module):
    """Compact peak-conditioned waveform-space CardioFlow network."""

    def __init__(self, width: int = 64, time_dim: int = 128,
                 peak_weight: float = 4.0, ppg_peak_weight: float = 1.0) -> None:
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError("width must be at least 8 and divisible by 8")
        if peak_weight < 0 or ppg_peak_weight < 0:
            raise ValueError("peak weights must be non-negative")
        self.width, self.time_dim = width, time_dim
        self.peak_weight, self.ppg_peak_weight = float(peak_weight), float(ppg_peak_weight)
        self.time_embedding = nn.Sequential(SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, 2 * time_dim),
                                             nn.SiLU(), nn.Linear(2 * time_dim, time_dim))
        self.input = nn.Conv1d(1, width, 7, padding=3)
        self.condition = nn.Conv1d(3, width, 7, padding=3)
        self.block0 = nn.ModuleList([ResidualBlock(width, time_dim), ResidualBlock(width, time_dim)])
        self.down1, self.cond1 = nn.Conv1d(width, width * 2, 5, stride=5, padding=2), nn.Conv1d(width, width * 2, 5, stride=5, padding=2)
        self.block1 = nn.ModuleList([ResidualBlock(width * 2, time_dim), ResidualBlock(width * 2, time_dim)])
        self.down2, self.cond2 = nn.Conv1d(width * 2, width * 4, 5, stride=5, padding=2), nn.Conv1d(width * 2, width * 4, 5, stride=5, padding=2)
        self.block2 = nn.ModuleList([ResidualBlock(width * 4, time_dim), ResidualBlock(width * 4, time_dim)])
        self.up1 = nn.Conv1d(width * 4, width * 2, 3, padding=1)
        self.up_block1 = nn.ModuleList([ResidualBlock(width * 2, time_dim), ResidualBlock(width * 2, time_dim)])
        self.up2 = nn.Conv1d(width * 2, width, 3, padding=1)
        self.up_block2 = nn.ModuleList([ResidualBlock(width, time_dim), ResidualBlock(width, time_dim)])
        self.output = nn.Sequential(nn.GroupNorm(_groups(width), width), nn.SiLU(), nn.Conv1d(width, 1, 7, padding=3))

    @staticmethod
    def _condition(ppg: Tensor) -> Tensor:
        ppg = ppg.transpose(1, 2) if ppg.ndim == 3 else ppg[:, None]
        difference = F.pad(ppg.diff(dim=-1), (1, 0))
        mask = peak_mask(ppg.transpose(1, 2)).transpose(1, 2)
        return torch.cat((ppg, difference, mask), dim=1)

    def forward(self, state: Tensor, time: Tensor, ppg: Tensor) -> Tensor:
        if state.ndim != 3 or ppg.ndim != 3 or state.shape != ppg.shape:
            raise ValueError(f"state and ppg must both be [B,L,1], got {state.shape} and {ppg.shape}")
        t, cond = self.time_embedding(time), self._condition(ppg)
        cond0 = self.condition(cond)
        h = self.input(state.transpose(1, 2)) + cond0
        for block in self.block0: h = block(h, t)
        skip0 = h
        cond1 = self.cond1(cond0); h = self.down1(h) + cond1
        for block in self.block1: h = block(h, t)
        skip1 = h
        h = self.down2(h) + self.cond2(cond1)
        for block in self.block2: h = block(h, t)
        h = F.interpolate(h, size=skip1.shape[-1], mode="linear", align_corners=False)
        h = self.up1(h) + skip1
        for block in self.up_block1: h = block(h, t)
        h = F.interpolate(h, size=skip0.shape[-1], mode="linear", align_corners=False)
        h = self.up2(h) + skip0
        for block in self.up_block2: h = block(h, t)
        return self.output(h).transpose(1, 2)

    def flow_matching_loss(self, ppg: Tensor, ecg: Tensor) -> Tensor:
        if ppg.shape != ecg.shape:
            raise ValueError("ppg and ecg must have identical shape")
        batch = ecg.shape[0]
        time = torch.rand(batch, device=ecg.device)
        source = torch.randn_like(ecg)
        state = (1 - time[:, None, None]) * source + time[:, None, None] * ecg
        velocity = ecg - source
        prediction = self(state, time, ppg)
        ecg_mask = peak_mask(ecg)
        ppg_mask = peak_mask(ppg)
        weights = 1.0 + self.peak_weight * ecg_mask + self.ppg_peak_weight * ppg_mask
        error = (prediction - velocity).square()
        return (error * weights).sum() / weights.sum().clamp_min(1.0)


class CardioFlowSampler:
    def __init__(self, model: CardioFlow, steps: int = 10) -> None:
        if steps != 10:
            raise ValueError("CardioFlow baseline is defined with T=10 Euler steps")
        self.model, self.steps = model, steps

    @torch.no_grad()
    def __call__(self, ppg: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if ppg.ndim != 3 or ppg.shape[-1] != 1:
            raise ValueError("ppg must have shape [B,1250,1]")
        state = torch.randn_like(ppg) if noise is None else noise.clone()
        if state.shape != ppg.shape:
            raise ValueError("noise and ppg must have the same shape")
        dt = 1.0 / self.steps
        for step in range(self.steps):
            time = ppg.new_full((len(ppg),), step * dt)
            state = state + dt * self.model(state, time, ppg)
        return state
