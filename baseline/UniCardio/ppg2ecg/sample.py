"""Generate ECG from PPG with a trained UniCardio and write the sample files
that ``v0/evaluation/run_eval.py`` expects.

Output layout matches ``v0/engine/solver.py::sample_shift`` exactly:
    overall_fake_data.npy   (N, 1250, 1) float64
    overall_gt_data.npy     (N, 1250, 1) float64
    overall_gt_ppg_data.npy (N, 1250, 1) float64
Test samples are visited in ``test.pt`` order (shuffle=False), so row i here is
row i of the PPGFlowECG baseline/personal outputs.

Example:
    python ppg2ecg/sample.py --ckpt results/unicardio/checkpoints/final.pt \
        --indices ppg2ecg/eval_subset_20k.npy
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "base_model")
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)

from personal_ppg2ecg.baseline.UniCardio.ppg2ecg.data import UniCardioPPGECG  # noqa: E402
from personal_ppg2ecg.baseline.UniCardio.ppg2ecg.model_ppg2ecg import UniCardioPPG2ECG  # noqa: E402

DEFAULT_DATA = "/root/personal_ppg2ecg/mimic-iv-aligned-ppg_ecgII-processed-filtered"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DEFAULT_DATA)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default=os.path.join(BASE, "base_no_compress_original.yaml"))
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(HERE), "results/unicardio/samples"))
    p.add_argument("--indices", default="", help="npy with test-set row indices; empty = full test set")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--sample_steps", type=int, default=6, help="DDIM steps (upstream test_final.py uses 6)")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--ddim", type=int, default=1, help="1 = DDIM, 0 = full 50-step ancestral DDPM")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ds = UniCardioPPGECG(args.data_dir, train=False, corrupt=False)
    unit_length = ds.length
    if args.indices:
        idx = np.load(args.indices)
        np.save(os.path.join(args.out_dir, "subset_indices.npy"), idx)
        view = Subset(ds, idx.tolist())
    else:
        view = ds
    loader = DataLoader(view, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    state = torch.load(args.ckpt, map_location=device)
    model = UniCardioPPG2ECG(config, device, L=unit_length * 4,
                             task_mode=state.get("task_mode", "translation")).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {args.ckpt} (step {state.get('step')}), N={len(view)}", flush=True)

    n = len(view)
    q = unit_length
    fake = np.empty((n, q, 1), dtype=np.float64)
    gt = np.empty((n, q, 1), dtype=np.float64)
    gt_ppg = np.empty((n, q, 1), dtype=np.float64)

    off, t0 = 0, time.time()
    with torch.no_grad():
        for bi, (sig, _, _, _) in enumerate(loader):
            sig = sig.to(device, non_blocking=True)
            b = sig.shape[0]
            ppg = sig[:, :, 0:q]
            ecg = sig[:, :, 2 * q:3 * q]
            zeros = torch.zeros_like(ppg)
            obs = torch.cat([ppg, zeros, zeros, zeros], dim=-1)

            out = model.generate(obs, n_samples=args.n_samples, model_flag="02",
                                 borrow_mode=2, sample_steps=args.sample_steps,
                                 DDIM_flag=args.ddim)
            if isinstance(out, tuple):  # DDIM path also returns the trajectory
                out = out[0]
            pred = out.mean(dim=1) if args.n_samples > 1 else out[:, 0]  # (B,1,q)

            fake[off:off + b] = pred[:, 0].double().cpu().numpy()[..., None]
            gt[off:off + b] = ecg[:, 0].double().cpu().numpy()[..., None]
            gt_ppg[off:off + b] = ppg[:, 0].double().cpu().numpy()[..., None]
            off += b
            if bi % 20 == 0:
                el = time.time() - t0
                print(f"{off}/{n} {el / max(off, 1) * n / 60:.1f} min total est.", flush=True)

    np.save(os.path.join(args.out_dir, "overall_fake_data.npy"), fake)
    np.save(os.path.join(args.out_dir, "overall_gt_data.npy"), gt)
    np.save(os.path.join(args.out_dir, "overall_gt_ppg_data.npy"), gt_ppg)
    print(f"saved {n} samples to {args.out_dir} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
