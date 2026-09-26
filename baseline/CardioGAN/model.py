"""PyTorch CardioGAN components for long (1250 sample) PPG/ECG windows."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, skip_channels, gate_channels, inter_channels):
        super().__init__()
        self.theta = nn.Conv1d(skip_channels, inter_channels, 1, bias=False)
        self.phi = nn.Conv1d(gate_channels, inter_channels, 1, bias=False)
        self.psi = nn.Conv1d(inter_channels, 1, 1)
    def forward(self, skip, gate):
        gate = F.interpolate(gate, size=skip.shape[-1], mode='linear', align_corners=False)
        score = F.relu(self.theta(skip) + self.phi(gate), inplace=True)
        return skip * torch.sigmoid(self.psi(score))

class AttentionUNet1D(nn.Module):
    """Attention U-Net generator with exact input/output length."""
    def __init__(self, base_channels=32, depth=6):
        super().__init__()
        if depth < 3: raise ValueError('depth must be at least 3')
        channels = [base_channels * min(2 ** i, 8) for i in range(depth)]
        self.down, self.down_norm = nn.ModuleList(), nn.ModuleList()
        in_ch = 1
        for i, out_ch in enumerate(channels):
            self.down.append(nn.Conv1d(in_ch, out_ch, 16, stride=2, padding=7, bias=False))
            self.down_norm.append(nn.Identity() if i == 0 else nn.InstanceNorm1d(out_ch, affine=True))
            in_ch = out_ch
        self.up, self.up_norm, self.gates = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        cur = channels[-1]
        for i in range(depth - 1, -1, -1):
            out_ch = channels[i]
            self.up.append(nn.ConvTranspose1d(cur, out_ch, 16, stride=2, padding=7, bias=False))
            self.up_norm.append(nn.InstanceNorm1d(out_ch, affine=True))
            self.gates.append(AttentionGate(out_ch, out_ch, max(out_ch // 2, 1)))
            cur = out_ch
        self.out = nn.Conv1d(cur, 1, 7, padding=3)
    def forward(self, x):
        length, skips, h = x.shape[-1], [], x
        for conv, norm in zip(self.down, self.down_norm):
            h = F.leaky_relu(norm(conv(h)), 0.2, inplace=True); skips.append(h)
        for i, (conv, norm, gate) in enumerate(zip(self.up, self.up_norm, self.gates)):
            h = F.relu(norm(conv(h)), inplace=True); skip = skips[-1 - i]
            if h.shape[-1] != skip.shape[-1]: h = F.interpolate(h, size=skip.shape[-1], mode='linear', align_corners=False)
            h = h + gate(skip, h)
        h = self.out(h)
        if h.shape[-1] != length: h = F.interpolate(h, size=length, mode='linear', align_corners=False)
        return torch.tanh(h)

class TimeDiscriminator(nn.Module):
    def __init__(self, base_channels=32, depth=5):
        super().__init__(); layers=[]; in_ch=1
        for i in range(depth):
            out_ch=min(base_channels*(2**i), base_channels*8)
            layers += [nn.Conv1d(in_ch,out_ch,16,stride=2,padding=7), nn.LeakyReLU(.2,inplace=True)]; in_ch=out_ch
        layers.append(nn.Conv1d(in_ch,1,3,padding=1)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class SpectrogramDiscriminator(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__(); c=base_channels
        self.net=nn.Sequential(nn.Conv2d(1,c,4,2,1),nn.LeakyReLU(.2,inplace=True),nn.Conv2d(c,c*2,4,2,1),nn.InstanceNorm2d(c*2,affine=True),nn.LeakyReLU(.2,inplace=True),nn.Conv2d(c*2,c*4,4,2,1),nn.InstanceNorm2d(c*4,affine=True),nn.LeakyReLU(.2,inplace=True),nn.Conv2d(c*4,c*8,4,2,1),nn.InstanceNorm2d(c*8,affine=True),nn.LeakyReLU(.2,inplace=True),nn.Conv2d(c*8,1,3,1,1))
    @staticmethod
    def transform(x):
        n_fft=min(256,x.shape[-1]); hop=max(n_fft//4,1); window=torch.hann_window(n_fft,device=x.device,dtype=x.dtype)
        spec=torch.stft(x.squeeze(1),n_fft=n_fft,hop_length=hop,win_length=n_fft,window=window,return_complex=True,center=True).abs()
        return torch.log(spec+1e-5).unsqueeze(1)
    def forward(self,x): return self.net(self.transform(x))

class CardioGAN(nn.Module):
    def __init__(self, base_channels=32, depth=6):
        super().__init__()
        self.g_ecg=AttentionUNet1D(base_channels,depth); self.g_ppg=AttentionUNet1D(base_channels,depth)
        self.d_ecg_t=TimeDiscriminator(base_channels); self.d_ppg_t=TimeDiscriminator(base_channels)
        self.d_ecg_f=SpectrogramDiscriminator(base_channels); self.d_ppg_f=SpectrogramDiscriminator(base_channels)
