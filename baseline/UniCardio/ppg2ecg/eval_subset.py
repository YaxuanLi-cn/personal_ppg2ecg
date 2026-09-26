"""Evaluate any samples dir on a fixed subset of test rows, using v0's metrics.

This imports ``v0/evaluation/run_eval.py`` unchanged, so MAE / RMSE / FD /
MAE_hr / FID are computed by exactly the same code that scores PPGFlowECG.
Pass the same ``--indices`` file for every method to keep the comparison on an
identical set of rows.

Example:
    python ppg2ecg/eval_subset.py --samples_dir results/unicardio/samples
    python ppg2ecg/eval_subset.py \
        --samples_dir /root/personal_ppg2ecg/v0/results/rectified_flow_baseline/mimic-iv-waveform/samples \
        --indices ppg2ecg/eval_subset_20k.npy
"""
import argparse
import os
import sys
import tempfile

import numpy as np

V0 = "/root/personal_ppg2ecg/v0"
sys.path.insert(0, V0)
from evaluation.run_eval import evaluate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples_dir", required=True)
    p.add_argument("--indices", default="",
                   help="npy of test-set row indices; omit if the dir is already the subset")
    p.add_argument("--sampling_rate", type=int, default=125)
    p.add_argument("--ecgfounder_ckpt", default="")
    p.add_argument("--no_normalize", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.indices:
        evaluate(args.samples_dir, sampling_rate=args.sampling_rate,
                 normalize=not args.no_normalize, ecgfounder_ckpt=args.ecgfounder_ckpt)
        return

    idx = np.load(args.indices)
    own = os.path.join(args.samples_dir, "subset_indices.npy")
    if os.path.exists(own) and np.array_equal(np.load(own), idx):
        # already generated on exactly these rows
        evaluate(args.samples_dir, sampling_rate=args.sampling_rate,
                 normalize=not args.no_normalize, ecgfounder_ckpt=args.ecgfounder_ckpt)
        return

    with tempfile.TemporaryDirectory(prefix="evalsub_") as tmp:
        for name in ("overall_gt_data.npy", "overall_fake_data.npy"):
            a = np.load(os.path.join(args.samples_dir, name), mmap_mode="r")
            np.save(os.path.join(tmp, name), np.asarray(a[idx]))
        print(f"[subset] {len(idx)} rows from {args.samples_dir}")
        evaluate(tmp, sampling_rate=args.sampling_rate,
                 normalize=not args.no_normalize, ecgfounder_ckpt=args.ecgfounder_ckpt)


if __name__ == "__main__":
    main()
