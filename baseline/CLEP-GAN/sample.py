"""Generate full-test PPG-to-ECG predictions with a selected CLEP-GAN model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import MIMICPairDataset
from model import CLEPGAN


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-dir", default="../../mimic-iv-aligned-ppg_ecgII-processed-filtered")
    parser.add_argument("--out-dir", default="results/clepgan/test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.out_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if state.get("selection", {}).get("test_used") is not False:
        raise ValueError("checkpoint must be selected without test-set information")
    cfg = state["args"]
    model = CLEPGAN(cfg["base_channels"], cfg["depth"], cfg["kernel_size"],
                    cfg["discriminator_channels"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    torch.set_num_threads(6)
    dataset = MIMICPairDataset(args.data_dir, "test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    shape = (len(dataset), 1250, 1)
    fake = np.lib.format.open_memmap(output / "overall_fake_data.npy", mode="w+",
                                     dtype=np.float32, shape=shape)
    gt = np.lib.format.open_memmap(output / "overall_gt_data.npy", mode="w+",
                                   dtype=np.float32, shape=shape)
    ppg_all = np.lib.format.open_memmap(output / "overall_gt_ppg_data.npy", mode="w+",
                                        dtype=np.float32, shape=shape)
    targets = np.empty(len(dataset), dtype=np.int64)
    offset = 0
    with torch.inference_mode():
        for batch_index, (ppg, ecg, indices) in enumerate(loader):
            end = offset + len(ecg)
            prediction = model(ppg.to(device, non_blocking=True))[2]
            fake[offset:end] = prediction.squeeze(1).cpu().numpy()[..., None]
            gt[offset:end] = ecg.squeeze(1).numpy()[..., None]
            ppg_all[offset:end] = ppg.squeeze(1).numpy()[..., None]
            targets[offset:end] = indices.numpy()
            offset = end
            if batch_index % 100 == 0:
                print(f"Generated {offset}/{len(dataset)}", flush=True)
    for array in (fake, gt, ppg_all):
        array.flush()
    np.save(output / "target_indices.npy", targets)
    (output / "locked_model.json").write_text(json.dumps({
        "checkpoint": str(Path(args.ckpt).resolve()), "sha256": sha256(Path(args.ckpt)),
        "training_step": state["step"], "validation": state["validation"],
        "selection": state["selection"], "test_count": len(dataset),
        "sampling_rate": 125, "sample_length": 1250,
    }, indent=2))
    print(f"Saved {offset} predictions to {output}")


if __name__ == "__main__":
    main()
