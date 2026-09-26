"""Dataset adapter for the repository's normalized MIMIC-IV tensors."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class MIMICPairDataset(Dataset):
    def __init__(self,data_dir,split='train',indices=None):
        self.root=Path(data_dir)
        if split not in ('train','test'): raise ValueError('split must be train or test')
        data=torch.load(self.root/f'{split}.pt',map_location='cpu',weights_only=False,mmap=True)
        if data['PPG'].shape != data['ECG'].shape or tuple(data['PPG'].shape[1:]) != (1250,): raise ValueError('expected [N,1250] PPG/ECG tensors')
        with (self.root/'z_score_mean_std.json').open() as f: stats=json.load(f)['train']
        self.ppg,self.ecg=data['PPG'],data['ECG']; self.ppg_mean,self.ppg_std=float(stats['PPG']['mean']),float(stats['PPG']['std']); self.ecg_mean,self.ecg_std=float(stats['ECG']['mean']),float(stats['ECG']['std'])
        self.indices=np.arange(len(self.ppg),dtype=np.int64) if indices is None else np.asarray(indices,dtype=np.int64)
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        idx=int(self.indices[i]); ppg=(self.ppg[idx].float()-self.ppg_mean)/(self.ppg_std+1e-8); ecg=(self.ecg[idx].float()-self.ecg_mean)/(self.ecg_std+1e-8)
        return ppg.unsqueeze(0),ecg.unsqueeze(0),idx
