import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data import PairedDataset, load_data
from predict import ECGPredictor
from runtime import ROOT, output_path, seed_all, loader, source_manifest
from evaluation.run_eval import evaluate


BASELINES = {
    'v0': {'MAE': 0.5204303210674103, 'RMSE': 0.8752301770672651, 'FD': 7.5225,
           'MAE_hr_paired': 2.5212, 'MAE_hr_group': 0.7514},
    'v1': {'MAE': 0.5784960209497133, 'RMSE': 0.9859376285086372, 'FD': 0.4554,
           'MAE_hr_paired': 2.4546, 'MAE_hr_group': 0.5570},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='results/final_test')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    folder = output_path(args.output)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Test output directory must be new')
    source_manifest(verify=True)
    checkpoint = Path(args.checkpoint).resolve()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('selection', {}).get('test_used') is not False:
        raise ValueError('Expected a checkpoint selected exclusively on training validation')
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'locked_model.json').write_text(json.dumps({'checkpoint': str(checkpoint),
        'sha256': checkpoint_sha, 'selection': state['selection']}, indent=2))
    torch.set_num_threads(6)
    seed_all(42)
    data, stats = load_data('test')
    targets = np.arange(len(data['PPG']))
    dataset = PairedDataset(data, stats, targets, targets, seed=42)
    v2_folder = ROOT.parent / 'v2/results/paired_v2/test'
    if not np.array_equal(dataset.refs, np.load(v2_folder / 'reference_indices.npy')):
        raise ValueError('Reference draws differ from the recorded v2 experiment')
    baseline_gt = np.load(v2_folder / 'overall_gt_data.npy', mmap_mode='r')
    baseline_ppg = np.load(v2_folder / 'overall_gt_ppg_data.npy', mmap_mode='r')
    model = ECGPredictor(checkpoint)
    shape = (len(dataset), 1250, 1)
    if baseline_gt.shape != shape or baseline_ppg.shape != shape:
        raise ValueError('Test sample count differs from baseline')
    outputs = {key: np.lib.format.open_memmap(folder / f'overall_{key}_data.npy', mode='w+', dtype=np.float32, shape=shape)
               for key in ('gt', 'gt_ppg', 'fake')}
    np.save(folder / 'reference_indices.npy', dataset.refs)
    offset = 0
    for batch_index, (ppg, ecg, ppg_ref, ecg_ref) in enumerate(loader(dataset, args.batch_size, 4)):
        prediction = model(ppg, ppg_ref, ecg_ref).cpu().numpy()
        part = slice(offset, offset + len(ecg))
        if not np.array_equal(ecg.numpy(), baseline_gt[part]) or not np.array_equal(ppg.numpy(), baseline_ppg[part]):
            raise ValueError('Test GT/PPG order or normalization differs from baseline')
        outputs['gt'][part], outputs['gt_ppg'][part], outputs['fake'][part] = ecg.numpy(), ppg.numpy(), prediction
        offset += len(ecg)
        if batch_index % 100 == 0:
            print(f'Generated {offset}/{len(dataset)}', flush=True)
    if offset != len(dataset):
        raise RuntimeError('Incomplete test generation')
    for array in outputs.values():
        array.flush()
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_sha:
        raise RuntimeError('Selected checkpoint changed during evaluation')
    del model
    torch.cuda.empty_cache()
    result = evaluate(folder, workers=args.workers)
    previous = dict(BASELINES)
    previous['v2'] = json.loads((v2_folder / 'five_metrics.json').read_text())['metrics']
    best = {key: min(metrics[key] for metrics in previous.values()) for key in result}
    finite = all(value is not None and np.isfinite(value) for value in result.values())
    comparison = {'v3': result, 'previous': previous, 'historical_best': best,
                  'better_than_historical_best': {key: bool(finite and result[key] < value) for key, value in best.items()},
                  'all_five_better_than_v2': bool(finite and all(result[key] < previous['v2'][key] for key in result)),
                  'all_five_better_than_historical_best': bool(finite and all(result[key] < best[key] for key in result)),
                  'test_count': offset, 'same_gt_ppg_and_reference_draws_as_v2': True,
                  'v0_v1_reference_draws': 'not saved in historical experiments; cannot verify exact equality',
                  'checkpoint_sha256': checkpoint_sha}
    (folder / 'comparison.json').write_text(json.dumps(comparison, indent=2))
    source_manifest(verify=True)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    main()
