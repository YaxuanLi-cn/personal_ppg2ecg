from pathlib import Path

import numpy as np
import torch

from personal_ppg2ecg.v4.model import SharedPrivateModel
from personal_ppg2ecg.v4.pretrained import FrozenConditions
from personal_ppg2ecg.v4.runtime import fixed_noise


class ECGPredictor:
    """Target-free inference: current PPG + a same-subject reference PPG/ECG pair.

    mode='sample' integrates the conditional rectified flow residual from a
    deterministic per-window Gaussian draw; mode='mean' uses only the
    deterministic private predictor. Inputs are float tensors [B,1250,1]
    normalized with the existing training-set statistics.
    """

    def __init__(self, checkpoint, device=None, mode='sample', flow_steps=8, seed=42, residual_scale=1.0):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
        if state.get('selection', {}).get('test_used') is not False:
            raise ValueError('Checkpoint must be selected without test data')
        self.model = SharedPrivateModel(**state['config']['model']).to(self.device)
        self.model.load_state_dict(state['ema'], strict=True)
        self.model.requires_grad_(False).eval()
        self.conditions = FrozenConditions().to(self.device)
        if mode not in ('sample', 'mean') or residual_scale < 0:
            raise ValueError('Unknown inference mode or negative residual scale')
        self.mode, self.flow_steps, self.seed = mode, flow_steps, seed
        self.residual_scale = residual_scale
        self.training_step = state['step']

    @torch.no_grad()
    def __call__(self, ppg, ppg_ref, ecg_ref, indices=None):
        ppg = torch.as_tensor(ppg, dtype=torch.float32, device=self.device)
        ppg_ref = torch.as_tensor(ppg_ref, dtype=torch.float32, device=self.device)
        ecg_ref = torch.as_tensor(ecg_ref, dtype=torch.float32, device=self.device)
        anchor, patient = self.conditions.shared(ppg), self.conditions.patient(ppg_ref, ecg_ref)
        shared = self.model.shared_encoder(ppg, anchor)
        if self.mode == 'mean':
            private = self.model.private_predictor(shared, patient)
        else:
            if indices is None:
                raise ValueError('Sample mode requires window indices for deterministic noise')
            noise = fixed_noise(np.asarray(indices), self.model.private_dim, seed=self.seed).to(self.device)
            private = self.model.sample_private(shared, patient, noise, self.flow_steps, self.residual_scale)
        return self.model.decoder(shared, private).float().cpu()
