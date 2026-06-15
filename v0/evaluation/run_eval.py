"""
Parameterized evaluation for generated ECG vs ground-truth.

Reads the sampling outputs (overall_gt_data.npy / overall_fake_data.npy) produced
by main.py synthesis mode and reports MAE, RMSE, FD, MAE_hr, and (if the ECGFounder
checkpoint is available) FID.

Usage:
    python evaluation/run_eval.py --samples_dir <dir> [--sampling_rate 125] \
        [--ecgfounder_ckpt <path>] [--no_normalize]
"""
import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.calculate_metric import (
    calculate_mae, calculate_rmse, calculate_fd, calculate_fd_for_small_sample,
    zscore_per_sample, MAE_hr, calculate_FID_score, compute_representations_in_batches,
    ecg_bpm_array,
)


def load_pair(samples_dir):
    gt = np.load(os.path.join(samples_dir, "overall_gt_data.npy"))
    fake = np.load(os.path.join(samples_dir, "overall_fake_data.npy"))
    if gt.ndim == 2:
        gt = gt[..., None]
    if fake.ndim == 2:
        fake = fake[..., None]
    return gt.astype(np.float64), fake.astype(np.float64)


def try_load_ecgfounder(ckpt_path):
    """Return an ECGFounder feature extractor, or None if checkpoint missing."""
    import torch
    from evaluation.calculate_metric import Net1D
    import torch.nn as nn
    if not (ckpt_path and os.path.exists(ckpt_path)):
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net1D(in_channels=1, base_filters=64, ratio=1,
                  filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                  m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2,
                  groups_width=16, verbose=False, use_bn=True, use_do=False, n_classes=1000)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k: v for k, v in sd.items() if not k.startswith("dense.")}
    model.load_state_dict(sd, strict=False)
    model.dense = nn.Identity()
    model.to(device).eval()
    return model, device


def evaluate(samples_dir, sampling_rate=125, normalize=True, ecgfounder_ckpt=None):
    gt, fake = load_pair(samples_dir)
    print(f"\n===== {samples_dir} =====")
    print(f"GT shape {gt.shape} | Fake shape {fake.shape}")

    if normalize:
        gt_n = zscore_per_sample(gt)
        fake_n = zscore_per_sample(fake)
    else:
        gt_n, fake_n = gt, fake

    results = {}
    results["MAE"] = float(calculate_mae(gt_n, fake_n))
    results["RMSE"] = float(calculate_rmse(gt_n, fake_n))

    # Use the PCA-reduced FD for numerical stability (raw 1250-dim covariance
    # makes sqrtm unstable and can yield negative FD).
    fd_mean, fd_std = calculate_fd_for_small_sample(gt_n, fake_n, pca_dim=64, eps=1e-4, n_trials=3)
    results["FD"] = float(fd_mean)

    try:
        # Paired per-sample HR error (repo's original MAE_hr definition).
        results["MAE_hr_paired"] = float(MAE_hr(gt, fake, ecg_sampling_freq=sampling_rate, window_size=10))
        # Group-level mean-HR difference (matches the paper's reported MAE_HR magnitude).
        gb = np.array(ecg_bpm_array(gt, sampling_rate, 10))
        fb = np.array(ecg_bpm_array(fake, sampling_rate, 10, filter=True))
        gb = gb[gb > 0]; fb = fb[fb > 0]
        results["MAE_hr_group"] = float(abs(np.nanmean(gb) - np.nanmean(fb)))
    except Exception as e:
        results["MAE_hr_paired"] = None
        results["MAE_hr_group"] = None
        print(f"[WARN] MAE_hr failed: {e}")

    loaded = try_load_ecgfounder(ecgfounder_ckpt)
    if loaded is None:
        results["FID"] = None
        print("[INFO] ECGFounder checkpoint not found -> skipping FID.")
    else:
        model, device = loaded
        fake_rep = compute_representations_in_batches(model, fake.astype(np.float32), batch_size=128, device=device)
        real_rep = compute_representations_in_batches(model, gt.astype(np.float32), batch_size=128, device=device)
        # PCA-reduce the 1024-dim ECGFounder features for sqrtm numerical stability.
        results["FID"] = float(calculate_FID_score(fake_rep, real_rep, eps=1e-6, pca_dim=64))

    print("--- Metrics ---")
    for k, v in results.items():
        print(f"{k:8s}: {'N/A' if v is None else f'{v:.4f}'}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples_dir", type=str, required=True)
    p.add_argument("--sampling_rate", type=int, default=125)
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--ecgfounder_ckpt", type=str,
                   default="/root/autodl-tmp/personal_ppg2ecg/models/ECGFounder/1_lead_ECGFounder.pth")
    args = p.parse_args()
    evaluate(args.samples_dir, sampling_rate=args.sampling_rate,
             normalize=not args.no_normalize, ecgfounder_ckpt=args.ecgfounder_ckpt)
