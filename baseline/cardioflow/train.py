"""Train the Nambu et al. (2025) conditional CardioFlow baseline.

Example::

    /venv/ppg/bin/python -B train.py --output results/cardioflow_t10 \
        --steps 100000 --batch-size 64

The training objective is exactly the straight-path flow matching loss.  The
``--sampling-steps`` argument is intentionally absent: this baseline is
defined with ten Euler steps and that value is recorded in the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from .data import WaveformDataset
    from .model import CardioFlow
except ImportError:  # direct ``python train.py`` from this directory
    from data import WaveformDataset
    from model import CardioFlow


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.shadow.items()}


def checkpoint_state(model, ema, optimizer, step, args, loss):
    return {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "loss": float(loss),
        "config": {"width": args.width, "time_dim": args.time_dim, "peak_weight": args.peak_weight, "ppg_peak_weight": args.ppg_peak_weight},
        "protocol": {
            "name": "conditional_cardioflow",
            "paper": "Nambu, Kohjima and Yamamoto, CardioFlow (ICASSP 2025)",
            "source": "x0~N(0,I), x1=normalized ECG, xt=(1-t)x0+t*x1",
            "objective": "peak-weighted flow matching on straight noise-to-ECG paths",
            "conditioning": "PPG, first difference, and dilated local PPG peak mask",
            "peak_mask": {"width": 15, "threshold": 0.35, "dilation": 9,
                          "ecg_weight": args.peak_weight, "ppg_weight": args.ppg_peak_weight},
            "sampling": "Euler",
            "sampling_steps": 10,
            "data_rate_hz": 125,
            "window_length": 1250,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conditional CardioFlow (T=10)")
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-dir", type=Path,
                        default=repo_root / "mimic-iv-aligned-ppg_ecgII-processed-filtered")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--peak-weight", type=float, default=4.0)
    parser.add_argument("--ppg-peak-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", type=int,
                        help="Use only this many training windows for a smoke run")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("steps, batch-size and workers must be valid positive values")
    if args.save_every < 1 or args.log_every < 1:
        raise ValueError("save-every and log-every must be positive")
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"{output} is non-empty; use a new directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device(args.device if args.device != "auto" else "cpu"))

    dataset = WaveformDataset(args.data_dir, "train", max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        drop_last=len(dataset) >= args.batch_size,
                        persistent_workers=args.workers > 0)
    model = CardioFlow(args.width, args.time_dim, args.peak_weight, args.ppg_peak_weight).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * 0.05)
    ema = EMA(model, args.ema_decay)
    start_step = 0
    if args.resume:
        path = output / "last.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        ema.shadow = {name: value.to(device) for name, value in state["ema"].items()}
        start_step = int(state["step"])
        if start_step >= args.steps:
            raise ValueError("checkpoint already reached --steps")
        for _ in range(start_step):
            scheduler.step()

    print(json.dumps({"device": str(device), "train_windows": len(dataset),
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "start_step": start_step, "total_steps": args.steps}), flush=True)
    iterator = iter(loader)
    started = time.monotonic()
    last_loss = math.nan
    model.train()
    for step in range(start_step, args.steps):
        try:
            ppg, ecg, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            ppg, ecg, _ = next(iterator)
        ppg, ecg = ppg.to(device, non_blocking=True), ecg.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = model.flow_matching_loss(ppg, ecg)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(model)
        last_loss = float(loss.detach())
        step_number = step + 1
        if step_number % args.log_every == 0 or step_number == 1:
            print(json.dumps({"step": step_number, "loss": last_loss,
                              "lr": optimizer.param_groups[0]["lr"],
                              "seconds": time.monotonic() - started}), flush=True)
        if step_number % args.save_every == 0 or step_number == args.steps:
            state = checkpoint_state(model, ema, optimizer, step_number, args, last_loss)
            torch.save(state, output / f"checkpoint-{step_number}.pt")
            torch.save(state, output / "last.pt")
    (output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(json.dumps({"complete": True, "step": args.steps, "loss": last_loss,
                      "checkpoint": str(output / "last.pt")}), flush=True)


if __name__ == "__main__":
    main()
