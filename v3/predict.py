from pathlib import Path

import torch
from torch import nn

from personal_ppg2ecg.v3.calibration import transport_tensor
from personal_ppg2ecg.v3.paired_model import PairedECGHead
from personal_ppg2ecg.v3.pretrained import FrozenConditions


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
        self.transport_count = len(state.get('transport', []))
        self.strength = float(state.get('transport_strength', 0.0))
        for index, layer in enumerate(state.get('transport', [])):
            for key, value in layer.items():
                self.register_buffer(f'{key}_{index}', torch.as_tensor(value).float())
        self.to(self.device).eval().requires_grad_(False)
        self.training_step = state['step']

    @torch.no_grad()
    def forward(self, ppg, ppg_ref, ecg_ref):
        if ppg.shape != ppg_ref.shape or ppg.shape != ecg_ref.shape:
            raise ValueError('Expected matching [B,1250,1] current PPG and reference signals')
        self.eval()
        ppg, ppg_ref, ecg_ref = (x.to(self.device) for x in (ppg, ppg_ref, ecg_ref))
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            shared, patient = self.conditions(ppg, ppg_ref, ecg_ref)
            point, _ = self.head(ppg, shared, patient)
        if self.strength:
            layers = [(getattr(self, f'matrix_{i}'), getattr(self, f'source_mean_{i}'), getattr(self, f'target_mean_{i}'))
                      for i in range(self.transport_count)]
            point = transport_tensor(point.float(), layers, self.strength)
        return point.float()
