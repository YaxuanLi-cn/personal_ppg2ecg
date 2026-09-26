"""Verify that the UniCardio adapter uses the same paired tensors as v0."""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.join(ROOT, "base_model")
V0 = "/root/personal_ppg2ecg/v0"
sys.path[:0] = [HERE, BASE, V0]

from personal_ppg2ecg.baseline.UniCardio.ppg2ecg.data import UniCardioPPGECG  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/root/personal_ppg2ecg/mimic-iv-aligned-ppg_ecgII-processed-filtered",
    )
    parser.add_argument("--rows", type=int, default=16)
    args = parser.parse_args()

    for split, train in (("train", True), ("test", False)):
        adapted = UniCardioPPGECG(args.data_dir, train=train, corrupt=False)
        raw = torch.load(os.path.join(args.data_dir, f"{split}.pt"), map_location="cpu", weights_only=False)
        with open(os.path.join(args.data_dir, "z_score_mean_std.json")) as handle:
            stats = json.load(handle)["train"]
        if not torch.equal(adapted.file_name, raw["file_name"]):
            raise AssertionError(f"{split}: row order/subject IDs differ")
        if adapted.length != 1250:
            raise AssertionError(f"{split}: expected 1250-sample windows, got {adapted.length}")

        n = min(args.rows, len(adapted))
        for idx in np.linspace(0, len(adapted) - 1, n, dtype=np.int64):
            signal, _, _, _ = adapted[int(idx)]
            ppg = (raw["PPG"][idx].clone() - stats["PPG"]["mean"]) / (stats["PPG"]["std"] + 1e-8)
            ecg = (raw["ECG"][idx].clone() - stats["ECG"]["mean"]) / (stats["ECG"]["std"] + 1e-8)
            if not torch.equal(signal[0, :1250], ppg):
                raise AssertionError(f"{split}[{idx}]: PPG normalization/order differs from v0")
            if not torch.equal(signal[0, 2500:3750], ecg):
                raise AssertionError(f"{split}[{idx}]: ECG normalization/order differs from v0")
            if torch.count_nonzero(signal[0, 1250:2500]) or torch.count_nonzero(signal[0, 3750:]):
                raise AssertionError(f"{split}[{idx}]: absent BP/placeholder slots are not zero")
        print(f"{split}: {len(adapted)} rows, first/order-spread {n} rows match v0 exactly")


if __name__ == "__main__":
    main()
