"""Loader for the repository's normalized paired PPG/ECG tensors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset


class WaveformDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str = "train",
                 max_samples: int | None = None, normalize: bool = True) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        self.data_dir = Path(data_dir).expanduser().resolve()
        path = self.data_dir / f"{split}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            values = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            values = torch.load(path, map_location="cpu")
        for key in ("PPG", "ECG", "file_name"):
            if key not in values:
                raise KeyError(f"{path} does not contain {key!r}")
        ppg = torch.as_tensor(values["PPG"], dtype=torch.float32)
        ecg = torch.as_tensor(values["ECG"], dtype=torch.float32)
        subjects = torch.as_tensor(values["file_name"], dtype=torch.int64)
        if ppg.ndim != 2 or ecg.shape != ppg.shape or ppg.shape[1] != 1250:
            raise ValueError(f"expected [N,1250] PPG/ECG, got {tuple(ppg.shape)} and {tuple(ecg.shape)}")
        if subjects.ndim != 1 or len(subjects) != len(ppg):
            raise ValueError("file_name must contain one subject id per window")
        if not torch.isfinite(ppg).all() or not torch.isfinite(ecg).all():
            raise ValueError("dataset contains non-finite values")
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive")
            ppg, ecg, subjects = ppg[:max_samples], ecg[:max_samples], subjects[:max_samples]
        self.stats: Dict[str, Dict[str, float]] = {}
        if normalize:
            stats_path = self.data_dir / "z_score_mean_std.json"
            if not stats_path.is_file():
                raise FileNotFoundError(stats_path)
            with stats_path.open() as handle:
                raw = json.load(handle)
            self.stats = raw.get("train", raw)
            for name in ("PPG", "ECG"):
                if name not in self.stats:
                    raise KeyError(f"normalization statistics lack {name}")
                mean, std = float(self.stats[name]["mean"]), float(self.stats[name]["std"])
                if not torch.isfinite(torch.tensor([mean, std])).all() or std <= 0:
                    raise ValueError(f"invalid {name} normalization statistics")
                if name == "PPG":
                    ppg = (ppg - mean) / (std + 1e-8)
                else:
                    ecg = (ecg - mean) / (std + 1e-8)
        self.ppg, self.ecg, self.subjects = ppg.contiguous(), ecg.contiguous(), subjects.contiguous()

    def __len__(self) -> int:
        return len(self.ppg)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.ppg[index, :, None], self.ecg[index, :, None], self.subjects[index]
