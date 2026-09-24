import torch
from torch import nn
from torch.nn import functional as F

from metrics import zscore_tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(4, channels), nn.SiLU(),
            nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation),
            nn.GroupNorm(4, channels), nn.SiLU(),
            nn.Conv1d(channels, channels, 7, padding=3),
        )

    def forward(self, x):
        return x + self.net(x)


class PairedECGHead(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError('Width must be a positive multiple of four')
        self.width = width
        self.stem = nn.Conv1d(2, width, 15, padding=7)
        self.high = ResidualBlock(width)
        self.down1 = nn.Sequential(nn.Conv1d(width, 2 * width, 15, stride=5, padding=7), ResidualBlock(2 * width))
        self.down2 = nn.Sequential(nn.Conv1d(2 * width, 4 * width, 15, stride=5, padding=7), ResidualBlock(4 * width))
        self.shared_projection = nn.Conv1d(4, 4 * width, 1)
        self.patient_projection = nn.Sequential(nn.Linear(256, 4 * width), nn.SiLU(), nn.Linear(4 * width, 8 * width))
        self.middle = nn.Sequential(*(ResidualBlock(4 * width, dilation=d) for d in (1, 2, 4, 8)))
        self.up1 = nn.Sequential(nn.Conv1d(6 * width, 2 * width, 1), ResidualBlock(2 * width))
        self.up2 = nn.Sequential(nn.Conv1d(3 * width, width, 1), ResidualBlock(width), ResidualBlock(width))
        self.point = nn.Conv1d(width, 1, 7, padding=3)
        self.uncertainty = nn.Sequential(nn.Conv1d(width, width, 7, padding=3), nn.SiLU(), nn.Conv1d(width, 1, 1))

    def forward(self, ppg, shared, patient):
        if ppg.ndim != 3 or ppg.shape[1:] != (1250, 1):
            raise ValueError('PPG must be [B,1250,1]')
        if shared.shape != (len(ppg), 4, 50) or patient.shape != (len(ppg), 256):
            raise ValueError('Invalid shared or patient condition')
        signal = zscore_tensor(ppg).transpose(1, 2)
        derivative = F.pad(signal.diff(dim=-1), (1, 0))
        high = self.high(self.stem(torch.cat((signal, derivative), dim=1)))
        low = self.down1(high)
        middle = self.down2(low) + self.shared_projection(shared)
        scale, shift = self.patient_projection(patient).chunk(2, dim=1)
        middle = self.middle(middle * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1))
        features = self.up1(torch.cat((F.interpolate(middle, size=low.shape[-1], mode='linear', align_corners=False), low), dim=1))
        features = self.up2(torch.cat((F.interpolate(features, size=high.shape[-1], mode='linear', align_corners=False), high), dim=1))
        mean = zscore_tensor(self.point(features).transpose(1, 2))
        std = F.softplus(self.uncertainty(features.detach()).float()).transpose(1, 2) + 0.02
        return mean, std


def distribution_objective(point, scale, target, spectral_weight=2.0, stft_weight=0.2):
    target = zscore_tensor(target)
    mae = F.l1_loss(point, target)
    mse = F.mse_loss(point, target)
    slope = F.l1_loss(point.diff(dim=1), target.diff(dim=1))
    p, y = point.squeeze(-1).float(), target.squeeze(-1).float()
    power_p = torch.fft.rfft(p, norm='ortho').abs().square().mean(dim=0)
    power_y = torch.fft.rfft(y, norm='ortho').abs().square().mean(dim=0)
    spectrum = F.mse_loss((power_p + 1e-6).sqrt(), (power_y + 1e-6).sqrt())
    stft = point.new_zeros(())
    for size in (64, 128, 256):
        window = torch.hann_window(size, device=point.device)
        p_spec = torch.stft(p, size, hop_length=size // 4, window=window, return_complex=True).abs()
        y_spec = torch.stft(y, size, hop_length=size // 4, window=window, return_complex=True).abs()
        stft = stft + F.l1_loss(p_spec.log1p(), y_spec.log1p()) / 3
    residual = (target - point).detach()
    nll = (scale.log() + 0.5 * (residual / scale).square()).mean()
    loss = mse + 0.5 * mae + 0.15 * slope + spectral_weight * spectrum + stft_weight * stft + 0.05 * nll
    return {'loss': loss, 'mae': mae, 'mse': mse, 'slope': slope, 'spectrum': spectrum, 'stft': stft, 'nll': nll}
