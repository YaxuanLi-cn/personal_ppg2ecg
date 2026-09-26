"""Reader for the repository's preprocessed 125 Hz, 1250-point MIMIC pairs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MIMICPairDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        root = Path(data_dir)
        data = torch.load(root / f"{split}.pt", map_location="cpu", weights_only=False, mmap=True)
        if data["PPG"].shape != data["ECG"].shape or tuple(data["PPG"].shape[1:]) != (1250,):
            raise ValueError("expected paired [N,1250] MIMIC PPG/ECG tensors")
        with (root / "z_score_mean_std.json").open() as stream:
            stats = json.load(stream)["train"]
        self.ppg, self.ecg = data["PPG"], data["ECG"]
        self.ppg_mean, self.ppg_std = float(stats["PPG"]["mean"]), float(stats["PPG"]["std"])
        self.ecg_mean, self.ecg_std = float(stats["ECG"]["mean"]), float(stats["ECG"]["std"])

    def __len__(self) -> int:
        return len(self.ppg)

    def __getitem__(self, index: int):
        ppg = (self.ppg[index].float() - self.ppg_mean) / (self.ppg_std + 1e-8)
        ecg = (self.ecg[index].float() - self.ecg_mean) / (self.ecg_std + 1e-8)
        return ppg.unsqueeze(0), ecg.unsqueeze(0), int(index)
