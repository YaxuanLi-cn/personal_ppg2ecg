import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import PairedDataset, load_data
from predict import ECGPredictor
from runtime import ROOT, output_path, seed_all, loader, move
from evaluation.run_eval import evaluate

V2_SPLIT = ROOT.parent / 'v2/results/paired_v2/split.npz'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('sample', 'mean'), default='mean')
    parser.add_argument('--flow-steps', type=int, default=8)
    parser.add_argument('--residual-scale', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    folder = output_path(args.output)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Validation output directory must be new')
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    seed_all(42)
    data, stats = load_data('train')
    split = np.load(V2_SPLIT)
    dataset = PairedDataset(data, stats, split['validation_indices'], split['train_indices'], seed=43)
    model = ECGPredictor(args.checkpoint, mode=args.mode, flow_steps=args.flow_steps,
                         residual_scale=args.residual_scale)
    shape = (len(dataset), 1250, 1)
    outputs = {key: np.lib.format.open_memmap(folder / f'overall_{key}_data.npy', mode='w+',
               dtype=np.float32, shape=shape) for key in ('gt', 'fake')}
    np.save(folder / 'reference_indices.npy', dataset.refs)
    offset = 0
    for ppg, ecg, ppg_ref, ecg_ref, idx in loader(dataset, 512, 4):
        prediction = model(ppg, ppg_ref, ecg_ref, indices=idx.numpy()).numpy()
        part = slice(offset, offset + len(ecg))
        outputs['gt'][part], outputs['fake'][part] = ecg.numpy(), prediction
        offset += len(ecg)
    for array in outputs.values():
        array.flush()
    del model
    torch.cuda.empty_cache()
    result = evaluate(folder, workers=args.workers)
    print(json.dumps({'mode': args.mode, 'residual_scale': args.residual_scale,
                      'metrics': result}), flush=True)


if __name__ == '__main__':
    main()
