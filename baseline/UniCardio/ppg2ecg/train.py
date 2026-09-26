"""Train UniCardio on our MIMIC-IV PPG/ECG pairs (PPG -> ECG translation).

The optimiser, LR schedule, loss expression, diffusion schedule and model config
are the upstream ones (``base_no_compress_original.yaml`` +
``utils_together_original.train``).  Differences, all forced by our data, are
documented in ``README.md``.

Example:
    python ppg2ecg/train.py --max_hours 24 --task_mode translation
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

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
    p.add_argument("--save_dir", default=os.path.join(os.path.dirname(HERE), "results/unicardio"))
    p.add_argument("--config", default=os.path.join(BASE, "base_no_compress_original.yaml"))
    p.add_argument("--task_mode", default="translation", choices=["translation", "unified"])
    p.add_argument("--batch_size", type=int, default=16,
                   help="16 == upstream per-GPU batch, keeps their loss/lr scaling exact")
    p.add_argument("--total_steps", type=int, default=110000,
                   help="used for the LR milestones (upstream: 18%% and 75%% of training)")
    p.add_argument("--max_hours", type=float, default=24.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--resume", default="")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train_log.csv")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda")

    ds = UniCardioPPGECG(args.data_dir, train=True, corrupt=(args.task_mode == "unified"))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        pin_memory=True, num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)
    unit_length = ds.length
    print(f"train samples={len(ds)} unit_length={unit_length} task_mode={args.task_mode}", flush=True)

    model = UniCardioPPG2ECG(config, device, L=unit_length * 4, task_mode=args.task_mode).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)

    optimizer = Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    p1, p2 = int(0.18 * args.total_steps), int(0.75 * args.total_steps)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        print(f"resumed from {args.resume} at step {step}", flush=True)

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("step,loss,lr,elapsed_h\n")

    def save(tag):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step,
                    "unit_length": unit_length, "task_mode": args.task_mode},
                   os.path.join(ckpt_dir, f"{tag}.pt"))

    t0 = time.time()
    running, n_running = 0.0, 0
    model.train()
    stop = False
    while not stop:
        for batch in loader:
            sig, imp, noi, msk = [b.to(device, non_blocking=True) for b in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = model(sig, sig_impute=imp, sig_denoise=noi, mask=msk,
                         task_dice=np.random.rand(1), dirty_dice=np.random.rand(1),
                         condition_dice=np.random.rand(1), train_threshold=0.0,
                         stage=1, is_train=1, train_gen_flag=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            running += loss.item()
            n_running += 1

            if step % args.log_every == 0:
                elapsed = (time.time() - t0) / 3600
                avg = running / max(n_running, 1)
                with open(log_path, "a") as f:
                    f.write(f"{step},{avg:.6f},{scheduler.get_last_lr()[0]:.3e},{elapsed:.4f}\n")
                print(f"step {step} loss {avg:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"elapsed {elapsed:.2f}h", flush=True)
                running, n_running = 0.0, 0

            if step % args.ckpt_every == 0:
                save(f"step-{step}")
                save("last")

            if (time.time() - t0) / 3600 >= args.max_hours or step >= args.total_steps:
                stop = True
                break

    save("last")
    save("final")
    print(f"done at step {step}, {(time.time() - t0) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
