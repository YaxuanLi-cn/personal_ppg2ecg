from pathlib import Path

import torch
from torch import nn

from personal_ppg2ecg.v2.paired_model import PairedECGHead
from personal_ppg2ecg.v2.pretrained import FrozenConditions


class ECGPredictor(nn.Module):
    def __init__(self, checkpoint, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        state = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
        config = state['config']
        self.amp = config['amp'] and self.device.type == 'cuda'
        self.head = PairedECGHead(config['width'])
        self.head.load_state_dict(state['ema'], strict=True)
        self.conditions = FrozenConditions(config['shared_checkpoint'], config['patient_checkpoint'])
        self.to(self.device).eval().requires_grad_(False)
        self.training_step = state['step']
        self.validation_metrics = state['best_validation']

    @torch.no_grad()
    def forward(self, ppg, ppg_ref, ecg_ref):
        if ppg.shape != ppg_ref.shape or ppg.shape != ecg_ref.shape:
            raise ValueError('Expected matching [B,1250,1] current PPG and reference signals')
        self.eval()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            shared, patient = self.conditions(ppg.to(self.device), ppg_ref.to(self.device), ecg_ref.to(self.device))
            point, std = self.head(ppg.to(self.device), shared, patient)
        return point.float(), std.float()
