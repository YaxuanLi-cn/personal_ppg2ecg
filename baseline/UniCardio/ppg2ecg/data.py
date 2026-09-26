"""Dataset adapter: our MIMIC-IV PPG/ECG pairs -> UniCardio's 4-slot token layout.

UniCardio's base model consumes a single 1-D tensor of length ``4 * unit_length``
holding four concatenated modality slots ``[A | B | C | D]`` where A/B/C are
PPG / BP / ECG and D is a placeholder used by the self-conditioning tasks.

Our dataset has no BP channel, so slot B is filled with zeros and never used as
a condition or a target (see ``train.py``).  Everything else -- the waveforms,
the train/test split and the z-score normalisation -- is byte-for-byte the same
data that ``v0/utils/ppgecg_dataset.py`` feeds to PPGFlowECG.
"""
import contextlib
import io
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# Upstream corruption helpers, used verbatim.
from self_process import AddNoise, imputation_pattern

PPG_SLOT, BP_SLOT, ECG_SLOT = 0, 1, 2

_SINK = io.StringIO()  # upstream helpers print on every call; silence them.


def load_zscore(data_dir):
    with open(os.path.join(data_dir, "z_score_mean_std.json"), "r") as f:
        return json.load(f)["train"]


class UniCardioPPGECG(Dataset):
    """Yields ``(signal, signal_impute, signal_noisy, mask)`` exactly as the
    upstream ``CustomSignalDataset`` does, but built from our ``train.pt`` /
    ``test.pt`` tensors.

    Shapes: signal/impute/noisy ``[1, 4*L]``, mask ``[1, L]`` with ``L = 1250``.
    """

    def __init__(self, data_dir, train=True, corrupt=True, snr=15):
        path = os.path.join(data_dir, "train.pt" if train else "test.pt")
        data = torch.load(path, map_location="cpu")
        self.ppg, self.ecg, self.file_name = data["PPG"], data["ECG"], data["file_name"]
        self.z = load_zscore(data_dir)
        self.corrupt = corrupt
        self.snr = snr
        self.length = self.ppg.shape[1]

    def __len__(self):
        return self.ppg.shape[0]

    def _norm(self, x, key):
        return (x - self.z[key]["mean"]) / (self.z[key]["std"] + 1e-8)

    def __getitem__(self, idx):
        ppg = self._norm(self.ppg[idx], "PPG")
        ecg = self._norm(self.ecg[idx], "ECG")
        L = self.length

        # [1, 3, L] stack in UniCardio channel order (PPG, BP, ECG); BP is absent.
        tri = np.zeros((1, 3, L), dtype=np.float32)
        tri[0, PPG_SLOT] = ppg.numpy()
        tri[0, ECG_SLOT] = ecg.numpy()

        if self.corrupt:
            with contextlib.redirect_stdout(_SINK):
                imputed, mask = imputation_pattern(tri.copy(), extended=True)
                noisy = AddNoise(tri.copy(), SNR=self.snr)
            imputed = imputed.reshape(3 * L).float()
            noisy = noisy.reshape(3 * L).float()
            mask = mask[0, 0].float()
        else:
            imputed = torch.zeros(3 * L)
            noisy = torch.zeros(3 * L)
            mask = torch.zeros(L)

        signal = torch.from_numpy(tri.reshape(3 * L))
        null = torch.zeros(L)
        return (
            torch.cat([signal, null])[None, :],
            torch.cat([imputed, null])[None, :],
            torch.cat([noisy, null])[None, :],
            mask[None, :],
        )
