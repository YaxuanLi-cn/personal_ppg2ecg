import math

import torch
from torch import nn
from torch.nn import functional as F

from metrics import zscore_tensor


class Block(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(4, channels), nn.SiLU(),
            nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation),
            nn.GroupNorm(4, channels), nn.SiLU(), nn.Conv1d(channels, channels, 7, padding=3))

    def forward(self, x):
        return x + self.net(x)


class SharedEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.signal = nn.Sequential(nn.Conv1d(2, width, 15, stride=5, padding=7), Block(width), Block(width, 2))
        self.anchor = nn.Conv1d(4, width, 1)
        self.finish = nn.Sequential(Block(width, 4), nn.GroupNorm(4, width))

    def forward(self, signal, anchor):
        signal = zscore_tensor(signal).transpose(1, 2)
        features = self.signal(torch.cat((signal, F.pad(signal.diff(dim=-1), (1, 0))), dim=1))
        return self.finish(features + F.interpolate(self.anchor(anchor), size=features.shape[-1], mode='linear', align_corners=False))


class PrivateEmbedding(nn.Module):
    def __init__(self, width, private_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, width, 15, stride=5, padding=7),
            Block(width), Block(width, 2), nn.GroupNorm(4, width), nn.SiLU(),
            nn.Conv1d(width, private_dim, 1), nn.Tanh())

    def forward(self, ecg):
        return self.net(zscore_tensor(ecg).transpose(1, 2))


class JointDecoder(nn.Module):
    def __init__(self, width, private_dim):
        super().__init__()
        self.private_dim = private_dim
        self.fusion = nn.Conv1d(width + private_dim, width, 1)
        self.low = nn.Sequential(Block(width), Block(width, 2), Block(width, 4))
        self.high = nn.Sequential(nn.Conv1d(width, width // 2, 7, padding=3),
            Block(width // 2), Block(width // 2), nn.Conv1d(width // 2, 1, 7, padding=3))

    def forward(self, shared, private):
        if private.shape[1] != self.private_dim or shared.shape[::2] != private.shape[::2]:
            raise ValueError('Decoder requires compatible shared and private embeddings')
        features = self.low(self.fusion(torch.cat((shared, private), dim=1)))
        wave = self.high(F.interpolate(features, size=1250, mode='linear', align_corners=False))
        return zscore_tensor(wave.transpose(1, 2))


class PrivatePredictor(nn.Module):
    def __init__(self, width, private_dim):
        super().__init__()
        self.patient = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width * 2))
        self.net = nn.Sequential(*(Block(width, d) for d in (1, 2, 4, 8)),
                                 nn.GroupNorm(4, width), nn.SiLU(), nn.Conv1d(width, private_dim, 1), nn.Tanh())

    def forward(self, shared, patient):
        scale, shift = self.patient(patient).chunk(2, dim=-1)
        return self.net(shared * (1 + scale[..., None]) + shift[..., None])


class PrivateFlow(nn.Module):
    def __init__(self, width, private_dim):
        super().__init__()
        self.state = nn.Conv1d(private_dim, width, 1)
        self.shared = nn.Conv1d(width, width, 1)
        self.time = nn.Sequential(nn.Linear(32, width), nn.SiLU(), nn.Linear(width, width))
        self.patient = nn.Linear(256, width * 2)
        self.blocks = nn.Sequential(*(Block(width, d) for d in (1, 2, 4, 8)))
        self.output = nn.Conv1d(width, private_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer('frequencies', torch.exp(torch.linspace(0, math.log(1000), 16)), persistent=False)

    def forward(self, state, time, shared, patient):
        phase = time.float()[:, None] * self.frequencies[None] * math.pi
        embedded = self.time(torch.cat((phase.sin(), phase.cos()), dim=-1))
        scale, shift = self.patient(patient).chunk(2, dim=-1)
        features = self.state(state) + self.shared(shared) + embedded[..., None]
        return self.output(self.blocks(features * (1 + scale[..., None]) + shift[..., None]))


def alignment_loss(shared_ppg, shared_ecg):
    p = F.normalize(shared_ppg.float(), dim=1)
    e = F.normalize(shared_ecg.float(), dim=1)
    return F.mse_loss(p, e)


def decorrelation_loss(shared, private):
    s, r = shared.float().mean(-1), private.float().mean(-1)
    s = F.normalize(s - s.mean(0, keepdim=True), dim=-1)
    r = F.normalize(r - r.mean(0, keepdim=True), dim=-1)
    return ((s.T @ r / len(s)) ** 2).sum()


def waveform_loss(prediction, target, spectral_weight=1.0):
    p, y = zscore_tensor(prediction).squeeze(-1), zscore_tensor(target).squeeze(-1)
    loss = F.mse_loss(p, y) + 0.5 * F.l1_loss(p, y) + 0.15 * F.l1_loss(p.diff(dim=-1), y.diff(dim=-1))
    if spectral_weight:
        sp = (torch.fft.rfft(p, norm='ortho').abs().square().mean(0) + 1e-6).sqrt()
        sy = (torch.fft.rfft(y, norm='ortho').abs().square().mean(0) + 1e-6).sqrt()
        spectral = 2 * F.mse_loss(sp, sy)
        for n in (64, 128, 256):
            window = torch.hann_window(n, device=p.device)
            a = torch.stft(p, n, hop_length=n // 4, window=window, return_complex=True).abs().log1p()
            b = torch.stft(y, n, hop_length=n // 4, window=window, return_complex=True).abs().log1p()
            spectral = spectral + (0.2 / 3) * F.l1_loss(a, b)
        loss = loss + spectral_weight * spectral
    return loss


class SharedPrivateModel(nn.Module):
    def __init__(self, width=64, private_dim=16, noise_scale=0.25):
        super().__init__()
        if width < 8 or width % 8 or private_dim < 1 or noise_scale <= 0:
            raise ValueError('Invalid latent configuration')
        self.width, self.private_dim, self.noise_scale = width, private_dim, noise_scale
        self.shared_encoder = SharedEmbedding(width)
        self.private_encoder = PrivateEmbedding(width // 2, private_dim)
        self.private_predictor = PrivatePredictor(width, private_dim)
        self.decoder = JointDecoder(width, private_dim)
        self.ppg_decoder = nn.Sequential(nn.Conv1d(width, width // 2, 1), Block(width // 2), nn.Conv1d(width // 2, 1, 7, padding=3))
        self.flow = PrivateFlow(width, private_dim)

    def configuration(self):
        return {'width': self.width, 'private_dim': self.private_dim, 'noise_scale': self.noise_scale}

    def representation_loss(self, ppg, ecg, anchor_ppg, anchor_ecg, patient, spectral_weight=1.0):
        shared_p = self.shared_encoder(ppg, anchor_ppg)
        shared_e = self.shared_encoder(ecg, anchor_ecg)
        private_e = self.private_encoder(ecg)
        predicted_private = self.private_predictor(shared_p, patient)
        predicted = self.decoder(shared_p, predicted_private)
        cross = self.decoder(shared_p, private_e)
        recon = self.decoder(shared_e, private_e)
        recon_ppg = self.ppg_decoder(F.interpolate(shared_p, size=1250, mode='linear', align_corners=False)).transpose(1, 2)
        terms = {
            'predicted_wave': waveform_loss(predicted, ecg, spectral_weight),
            'cross_reconstruction': waveform_loss(cross, ecg, 0),
            'self_reconstruction': waveform_loss(recon, ecg, 0),
            'ppg_reconstruction': F.mse_loss(zscore_tensor(recon_ppg), zscore_tensor(ppg)),
            'private_supervision': F.mse_loss(predicted_private.float(), private_e.detach().float()),
            'shared_alignment': alignment_loss(shared_p, shared_e),
            'decorrelation': decorrelation_loss(shared_e, private_e),
            'private_variance': F.relu(0.1 - private_e.float().std(dim=(0, 2), unbiased=False)).mean(),
        }
        weights = (1., 0.5, 0.25, 0.1, 0.1, 0.1, 0.01, 0.01)
        terms['loss'] = sum(w * term for w, term in zip(weights, terms.values()))
        return terms

    def freeze_representation(self):
        self.requires_grad_(False).eval()
        self.flow.requires_grad_(True).train()

    def sample_private(self, shared, patient, noise, steps=8, residual_scale=1.0):
        if steps < 1 or noise.shape != (len(shared), self.private_dim, shared.shape[-1]):
            raise ValueError('Invalid flow step count or private noise shape')
        mean = self.private_predictor(shared, patient)
        state = noise * self.noise_scale
        for step in range(steps):
            time = state.new_full((len(state),), step / steps)
            state = state + self.flow(state, time, shared, patient) / steps
        return mean + residual_scale * state

    def flow_loss(self, ppg, ecg, anchor_ppg, patient, rollout_steps=0):
        with torch.no_grad():
            shared = self.shared_encoder(ppg, anchor_ppg)
            mean = self.private_predictor(shared, patient)
            target = self.private_encoder(ecg) - mean
        noise = torch.randn_like(target) * self.noise_scale
        time = torch.rand(len(target), device=target.device)
        t = time[:, None, None]
        state = (1 - t) * noise + t * target
        velocity = self.flow(state, time, shared, patient)
        endpoint = mean + state + (1 - t) * velocity
        terms = {
            'flow_matching': F.mse_loss(velocity.float(), (target - noise).float()),
            'endpoint_wave': waveform_loss(self.decoder(shared, endpoint), ecg, 0.25),
        }
        loss = terms['flow_matching'] + 0.25 * terms['endpoint_wave']
        if rollout_steps:
            private = self.sample_private(shared, patient, torch.randn_like(target), rollout_steps)
            terms['rollout_wave'] = waveform_loss(self.decoder(shared, private), ecg)
            loss = loss + terms['rollout_wave']
        terms['loss'] = loss
        return terms

    def forward(self, ppg, anchor_ppg, patient, noise, steps=8, residual_scale=1.0):
        shared = self.shared_encoder(ppg, anchor_ppg)
        private = self.sample_private(shared, patient, noise, steps, residual_scale)
        return self.decoder(shared, private)
