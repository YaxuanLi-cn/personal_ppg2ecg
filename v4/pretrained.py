from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SHARED_CHECKPOINT = ROOT.parent / 'v1/results/cardioalign_shared_private/mimic-iv-waveform/checkpoints/VAE-iter-40000.pth'
PATIENT_CHECKPOINT = ROOT.parent / 'v1/results/IBExtractor/mimic-iv-waveform/checkpoints/IBExtractor-iter-40000.pth'


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv1d(in_channels, out_channels, 15, padding=7)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv1d(out_channels, out_channels, 15, padding=7)
        self.residual_layer = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv_1(F.silu(self.groupnorm_1(x)))
        out = self.conv_2(F.silu(self.groupnorm_2(out)))
        return out + self.residual_layer(x)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_proj = nn.Linear(channels, 3 * channels)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x):
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        attention = (q @ k.transpose(-1, -2) / x.shape[-1] ** 0.5).softmax(dim=-1)
        return self.out_proj(attention @ v)


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(channels)

    def forward(self, x):
        return x + self.attention(x.transpose(1, 2)).transpose(1, 2)


class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Conv1d(1, 128, 15, padding=7), VAE_ResidualBlock(128, 128), VAE_ResidualBlock(128, 128),
            nn.Conv1d(128, 128, 15, stride=5, padding=7), VAE_ResidualBlock(128, 256), VAE_ResidualBlock(256, 256),
            nn.Conv1d(256, 256, 15, padding=7), VAE_ResidualBlock(256, 512), VAE_ResidualBlock(512, 512),
            nn.Conv1d(512, 512, 17, stride=5, padding=8), VAE_ResidualBlock(512, 512), VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512), VAE_AttentionBlock(512), VAE_ResidualBlock(512, 512),
            nn.GroupNorm(32, 512), nn.SiLU(), nn.Conv1d(512, 8, 15, padding=7), nn.Conv1d(8, 8, 1),
        ])

    def forward(self, ppg):
        x = ppg.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        mean, _ = x.chunk(2, dim=1)
        return mean * 0.18215


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class IBEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.feed_forward = FeedForward()
        self.layer_norm1 = nn.LayerNorm(256)
        self.layer_norm2 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        normalized = self.layer_norm1(x)
        x = x + self.dropout(self.self_attn(normalized, normalized, normalized, need_weights=False)[0])
        return x + self.dropout(self.feed_forward(self.layer_norm2(x)))


class PatientEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.signal_embedding = nn.Linear(32, 256)
        self.down_sample = nn.Sequential(nn.Conv1d(2, 64, 15, stride=5, padding=7), nn.ReLU(),
                                         nn.Conv1d(64, 32, 15, stride=5, padding=7), nn.ReLU())
        self.layers = nn.ModuleList([IBEncoderLayer() for _ in range(3)])
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    def forward(self, ppg_ref, ecg_ref):
        x = self.signal_embedding(self.down_sample(torch.cat((ppg_ref, ecg_ref), dim=-1).transpose(1, 2)).transpose(1, 2))
        inv = 1 / (10000 ** (torch.arange(0, 256, 2, device=x.device).float() / 256))
        freqs = torch.arange(x.shape[1], device=x.device).float().unsqueeze(-1) * inv.unsqueeze(0)
        phase = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
        left, right = x.chunk(2, dim=-1)
        x = x * phase.cos() + torch.cat((-right, left), dim=-1) * phase.sin()
        for layer in self.layers:
            x = layer(x)
        return F.normalize(x.mean(dim=1), dim=-1)


class FrozenConditions(nn.Module):
    def __init__(self, shared_checkpoint=SHARED_CHECKPOINT, patient_checkpoint=PATIENT_CHECKPOINT):
        super().__init__()
        self.shared = SharedEncoder()
        self.patient = PatientEncoder()
        shared = torch.load(shared_checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if not shared.get('representation', {}).get('use_shared_private', False):
            raise ValueError('Expected v1 shared/private representation checkpoint')
        self.shared.load_state_dict(shared['encoder_ppg'], strict=True)
        patient = torch.load(patient_checkpoint, map_location='cpu', weights_only=False, mmap=True)
        self.patient.load_state_dict(patient['model'], strict=True)
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, ppg, ppg_ref, ecg_ref):
        self.eval()
        return self.shared(ppg).float(), self.patient(ppg_ref, ecg_ref).float()
