"""Train CLEP-GAN on the repository's existing paired MIMIC-IV windows."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data import MIMICPairDataset
from model import CLEPGAN


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model: CLEPGAN, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    abs_sum = square_sum = count = 0
    for ppg, ecg, _ in loader:
        ppg, ecg = ppg.to(device, non_blocking=True), ecg.to(device, non_blocking=True)
        fake = model(ppg, ecg)[2]
        diff = (fake - ecg).double()
        abs_sum += diff.abs().sum().item()
        square_sum += diff.square().sum().item()
        count += diff.numel()
    model.train()
    return {"MAE": abs_sum / count, "RMSE": (square_sum / count) ** 0.5}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../../mimic-iv-aligned-ppg_ecgII-processed-filtered")
    parser.add_argument("--out-dir", default="results/clepgan")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--kernel-size", type=int, default=31)
    parser.add_argument("--discriminator-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-gen", type=float, default=30.)
    parser.add_argument("--alpha-time", type=float, default=3.)
    parser.add_argument("--beta-freq", type=float, default=1.)
    parser.add_argument("--val-fraction", type=float, default=.05)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    if args.steps < 1 or args.val_every < 1 or not 0 < args.val_fraction < .5:
        parser.error("steps/val-every must be positive and val-fraction in (0, .5)")

    seed_all(args.seed)
    torch.set_num_threads(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = MIMICPairDataset(args.data_dir, "train")
    order = np.random.default_rng(args.seed).permutation(len(dataset))
    n_val = max(1, int(len(order) * args.val_fraction))
    val_indices, train_indices = order[:n_val], order[n_val:]
    val_indices = val_indices[:min(len(val_indices), args.val_samples)]
    train_loader = DataLoader(Subset(dataset, train_indices.tolist()), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(Subset(dataset, val_indices.tolist()), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=args.num_workers > 0)
    if not len(train_loader):
        raise ValueError("Training split is smaller than one batch")

    model = CLEPGAN(args.base_channels, args.depth, args.kernel_size,
                    args.discriminator_channels).to(device)
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.SmoothL1Loss()
    gen_parameters = (list(model.ppg_generator.parameters()) +
                      list(model.ecg_generator.parameters()) + [model.logit_scale])
    disc_parameters = (list(model.ecg_time_discriminator.parameters()) +
                       list(model.ecg_freq_discriminator.parameters()))
    opt_g = torch.optim.Adam(gen_parameters, lr=args.lr, betas=(.5, .999))
    opt_d = torch.optim.Adam(disc_parameters, lr=args.lr, betas=(.5, .999))
    start_step, best_score = 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if "opt_g" in state and "opt_d" in state:
            opt_g.load_state_dict(state["opt_g"])
            opt_d.load_state_dict(state["opt_d"])
        start_step = int(state["step"])
        best_score = float(state.get("best_score", best_score))

    log_path = output / "train_log.jsonl"
    train_iter = iter(train_loader)
    started = time.time()
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        try:
            ppg, ecg, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            ppg, ecg, _ = next(train_iter)
        ppg, ecg = ppg.to(device, non_blocking=True), ecg.to(device, non_blocking=True)
        ppg_recon, ecg_recon, fake_ecg, ppg_features, ecg_features = model(ppg, ecg)

        # Alternating ECG time- and frequency-domain discriminator updates.
        opt_d.zero_grad(set_to_none=True)
        real_t = model.ecg_time_discriminator(ecg)
        fake_t = model.ecg_time_discriminator(fake_ecg.detach())
        loss_dt = bce(real_t, torch.ones_like(real_t)) + bce(fake_t, torch.zeros_like(fake_t))
        real_f = model.ecg_freq_discriminator(ecg)
        fake_f = model.ecg_freq_discriminator(fake_ecg.detach())
        loss_df = bce(real_f, torch.ones_like(real_f)) + bce(fake_f, torch.zeros_like(fake_f))
        loss_d = args.alpha_time * loss_dt + args.beta_freq * loss_df
        loss_d.backward()
        opt_d.step()

        for parameter in disc_parameters:
            parameter.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        contrast = model.contrastive(ppg_features, ecg_features)
        rec_ppg, rec_ecg, rec_cross = l1(ppg_recon, ppg), l1(ecg_recon, ecg), l1(fake_ecg, ecg)
        adv_t_logits = model.ecg_time_discriminator(fake_ecg)
        adv_f_logits = model.ecg_freq_discriminator(fake_ecg)
        adv_t = bce(adv_t_logits, torch.ones_like(adv_t_logits))
        adv_f = bce(adv_f_logits, torch.ones_like(adv_f_logits))
        loss_gen = args.lambda_gen * (rec_ppg + rec_ecg + rec_cross + contrast) + \
            args.alpha_time * adv_t + args.beta_freq * adv_f
        loss_gen.backward()
        opt_g.step()
        for parameter in disc_parameters:
            parameter.requires_grad_(True)

        if step % 100 == 0 or step == 1:
            record = {"step": step, "loss_d": float(loss_d.detach()),
                      "loss_gen": float(loss_gen.detach()), "contrast": float(contrast.detach()),
                      "recon_ppg": float(rec_ppg.detach()), "recon_ecg": float(rec_ecg.detach()),
                      "recon_ppg_to_ecg": float(rec_cross.detach()),
                      "hours": (time.time() - started) / 3600}
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

        if step % args.val_every == 0 or step == args.steps:
            metrics = validate(model, val_loader, device)
            score = metrics["MAE"] + metrics["RMSE"]
            state = {"step": step, "model": model.state_dict(), "opt_g": opt_g.state_dict(),
                     "opt_d": opt_d.state_dict(), "args": vars(args),
                     "validation": metrics, "best_score": min(best_score, score),
                     "selection": {"criterion": "validation MAE + RMSE", "test_used": False,
                                   "train_indices": int(len(train_indices)),
                                   "validation_indices": int(len(val_indices)),
                                   "seed": args.seed}}
            torch.save(state, output / "last.pt")
            if score < best_score:
                best_score = score
                state["best_score"] = best_score
                torch.save(state, output / "best.pt")
            print(json.dumps({"step": step, "validation": metrics,
                              "best_score": best_score}), flush=True)

    final = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    final["args"] = vars(args)
    torch.save(final, output / "final.pt")


if __name__ == "__main__":
    main()
