"""Generate the test split with a trained T=10 CardioFlow checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .data import WaveformDataset
    from .model import CardioFlow, CardioFlowSampler
except ImportError:
    from data import WaveformDataset
    from model import CardioFlow, CardioFlowSampler


def fixed_noise(start: int, count: int, length: int, seed: int) -> torch.Tensor:
    """Make per-window Gaussian draws independent of batch size and workers."""
    values = []
    for index in range(start, start + count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        values.append(rng.standard_normal((length, 1), dtype=np.float32))
    return torch.from_numpy(np.stack(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample conditional CardioFlow (T=10)")
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-dir", type=Path,
                        default=repo_root / "mimic-iv-aligned-ppg_ecgII-processed-filtered")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", type=int,
                        help="Generate only a prefix, useful for smoke tests")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} is non-empty; use a new directory or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative")
    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device(args.device if args.device != "auto" else "cpu"))
    try:
        state = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu")
    if state.get("protocol", {}).get("sampling_steps") != 10:
        raise ValueError("checkpoint is not a T=10 CardioFlow checkpoint")
    config = state.get("config", {})
    model = CardioFlow(int(config.get("width", 64)), int(config.get("time_dim", 128)), float(config.get("peak_weight", 4.0)), float(config.get("ppg_peak_weight", 1.0))).to(device)
    weights = state.get("ema", state.get("model"))
    if weights is None:
        raise KeyError("checkpoint must contain model or ema weights")
    model.load_state_dict(weights, strict=True)
    model.eval()
    sampler = CardioFlowSampler(model, steps=10)
    dataset = WaveformDataset(args.data_dir, "test", max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    shape = (len(dataset), 1250, 1)
    fake = np.lib.format.open_memmap(output / "overall_fake_data.npy", mode="w+",
                                     dtype=np.float32, shape=shape)
    gt = np.lib.format.open_memmap(output / "overall_gt_data.npy", mode="w+",
                                   dtype=np.float32, shape=shape)
    ppg_out = np.lib.format.open_memmap(output / "overall_gt_ppg_data.npy", mode="w+",
                                        dtype=np.float32, shape=shape)
    offset = 0
    with torch.no_grad():
        for ppg, ecg, _ in loader:
            count = len(ppg)
            ppg_device = ppg.to(device, non_blocking=True)
            noise = fixed_noise(offset, count, ppg.shape[1], args.seed).to(device)
            prediction = sampler(ppg_device, noise=noise).cpu().numpy().astype(np.float32)
            sl = slice(offset, offset + count)
            fake[sl], gt[sl], ppg_out[sl] = prediction, ecg.numpy(), ppg.numpy()
            offset += count
            if offset % (args.batch_size * 10) == 0 or offset == len(dataset):
                print(f"Generated {offset}/{len(dataset)}", flush=True)
    fake.flush(), gt.flush(), ppg_out.flush()
    np.save(output / "target_indices.npy", np.arange(len(dataset), dtype=np.int64))
    metadata = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "sampling_steps": 10,
        "sampler": "Euler",
        "model": "CardioFlow peak-aware waveform flow",
        "seed": args.seed,
        "shape": list(shape),
        "normalized_with": str((Path(args.data_dir).resolve() / "z_score_mean_std.json")),
    }
    (output / "generation.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"complete": True, "output": str(output), "shape": list(shape)}), flush=True)


if __name__ == "__main__":
    main()
