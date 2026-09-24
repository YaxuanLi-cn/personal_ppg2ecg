import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data import PairedDataset, load_data
from predict import ECGPredictor
from runtime import ROOT, output_path, seed_all, loader

V2_SPLIT = ROOT.parent / 'v2/results/paired_v2/split.npz'


def collect(model, dataset, batch_size=512, workers=4):
    fake = np.empty((len(dataset), 1250, 1), dtype=np.float32)
    real = np.empty((len(dataset), 1250, 1), dtype=np.float32)
    offset = 0
    for ppg, ecg, ppg_ref, ecg_ref, idx in loader(dataset, batch_size, workers):
        prediction = model(ppg, ppg_ref, ecg_ref, indices=idx.numpy()).numpy()
        n = len(ecg)
        fake[offset:offset + n], real[offset:offset + n] = prediction, ecg.numpy()
        offset += n
    if offset != len(dataset):
        raise RuntimeError('Incomplete collection')
    return real, fake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('sample', 'mean'), default='mean')
    parser.add_argument('--residual-scale', type=float, default=1.0)
    parser.add_argument('--flow-steps', type=int, default=8)
    parser.add_argument('--calibration-size', type=int, default=65536)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    folder = output_path(args.output)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Calibration cache directory must be new')
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    seed_all(42)
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    data, stats = load_data('train')
    split = np.load(V2_SPLIT)
    fit, validation = split['train_indices'], split['validation_indices']
    rng = np.random.default_rng(args.seed)
    cal_idx = np.sort(rng.choice(fit, args.calibration_size, replace=False))
    model = ECGPredictor(checkpoint, mode=args.mode, flow_steps=args.flow_steps,
                       residual_scale=args.residual_scale)
    for name, targets in (('calibration', cal_idx), ('validation', validation)):
        dataset = PairedDataset(data, stats, targets, fit, seed=44)
        np.save(folder / f'{name}_indices.npy', targets)
        np.save(folder / f'{name}_reference_indices.npy', dataset.refs)
        real, fake = collect(model, dataset, args.workers)
        np.save(folder / f'{name}_real.npy', real)
        np.save(folder / f'{name}_fake.npy', fake)
        print(f'{name}: collected {len(real)} windows', flush=True)
    provenance = {'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
                  'data_source': 'train.pt only', 'mode': args.mode,
                  'residual_scale': args.residual_scale, 'flow_steps': args.flow_steps,
                  'calibration_size': int(args.calibration_size), 'seed': args.seed,
                  'validation_split': str(V2_SPLIT)}
    (folder / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
