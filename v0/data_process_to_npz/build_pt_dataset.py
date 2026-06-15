"""
Build train.pt / test.pt / z_score_mean_std.json from the per-subject filtered
.npz files produced by step2_fast.py.

This step is required by the dataset classes in utils/ppgecg_dataset.py
(PPGECGDataset / PPGECGPairDataset), which load:
    - <data_dir>/train.pt   {'PPG': FloatTensor[N,L], 'ECG': FloatTensor[N,L], 'file_name': LongTensor[N]}
    - <data_dir>/test.pt
    - <data_dir>/z_score_mean_std.json   {'train': {'PPG': {'mean','std'}, 'ECG': {'mean','std'}}}

Notes:
- file_name is stored as an int64 subject id (the 'p' prefix is stripped) because
  the dataset does int(subject_id) and groups samples by subject for the
  personalized (ppf) reference sampling.
- A per-subject train/test split is used so that every subject present in a split
  has at least 2 samples (needed by the ppf reference sampler). Subjects with
  fewer than `min_for_test` samples go entirely to train.
"""

import os
import json
import glob
import argparse
import numpy as np
import torch


def subject_to_int(name: str) -> int:
    s = str(name)
    if s and s[0].lower() == 'p':
        s = s[1:]
    return int(s)


def build(input_dir: str, test_ratio: float = 0.1, min_for_test: int = 10,
          min_per_subject: int = 2, seed: int = 42):
    rng = np.random.RandomState(seed)
    files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
    print(f"Found {len(files)} subject npz files in {input_dir}")

    train_ppg, train_ecg, train_fn = [], [], []
    test_ppg, test_ecg, test_fn = [], [], []
    n_subj_train = n_subj_test = 0

    for f in files:
        d = np.load(f, allow_pickle=True)
        ppg = d['PPG'].astype(np.float32)
        ecg = d['ECG'].astype(np.float32)
        names = d['file_name']
        n = ppg.shape[0]
        if n < min_per_subject:
            continue
        sid = subject_to_int(names[0])

        idx = rng.permutation(n)
        if n >= min_for_test:
            n_test = max(min_per_subject, int(round(test_ratio * n)))
            n_test = min(n_test, n - min_per_subject)  # keep >=min in train
        else:
            n_test = 0

        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        if len(train_idx) >= min_per_subject:
            train_ppg.append(ppg[train_idx]); train_ecg.append(ecg[train_idx])
            train_fn.append(np.full(len(train_idx), sid, dtype=np.int64))
            n_subj_train += 1
        if len(test_idx) >= min_per_subject:
            test_ppg.append(ppg[test_idx]); test_ecg.append(ecg[test_idx])
            test_fn.append(np.full(len(test_idx), sid, dtype=np.int64))
            n_subj_test += 1

    train_ppg = np.concatenate(train_ppg, axis=0)
    train_ecg = np.concatenate(train_ecg, axis=0)
    train_fn = np.concatenate(train_fn, axis=0)
    test_ppg = np.concatenate(test_ppg, axis=0) if test_ppg else np.empty((0, train_ppg.shape[1]), np.float32)
    test_ecg = np.concatenate(test_ecg, axis=0) if test_ecg else np.empty((0, train_ppg.shape[1]), np.float32)
    test_fn = np.concatenate(test_fn, axis=0) if test_fn else np.empty((0,), np.int64)

    print(f"Train: {train_ppg.shape[0]} segments from {n_subj_train} subjects")
    print(f"Test:  {test_ppg.shape[0]} segments from {n_subj_test} subjects")

    # z-score stats computed on TRAIN only (global scalar mean/std)
    z = {
        "train": {
            "PPG": {"mean": float(train_ppg.mean()), "std": float(train_ppg.std())},
            "ECG": {"mean": float(train_ecg.mean()), "std": float(train_ecg.std())},
        }
    }

    train_pt = {
        "PPG": torch.from_numpy(train_ppg),
        "ECG": torch.from_numpy(train_ecg),
        "file_name": torch.from_numpy(train_fn),
    }
    test_pt = {
        "PPG": torch.from_numpy(test_ppg),
        "ECG": torch.from_numpy(test_ecg),
        "file_name": torch.from_numpy(test_fn),
    }

    torch.save(train_pt, os.path.join(input_dir, "train.pt"))
    torch.save(test_pt, os.path.join(input_dir, "test.pt"))
    with open(os.path.join(input_dir, "z_score_mean_std.json"), "w") as fp:
        json.dump(z, fp, indent=2)

    print("Saved train.pt, test.pt, z_score_mean_std.json to", input_dir)
    print("z_score:", json.dumps(z, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True,
                   help="Directory containing per-subject filtered .npz files")
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--min_for_test", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build(args.input, test_ratio=args.test_ratio, min_for_test=args.min_for_test, seed=args.seed)
