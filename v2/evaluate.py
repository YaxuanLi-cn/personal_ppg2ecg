import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import PairedDataset, load_data
from metrics import paired_sums, finish_metrics, zscore_numpy
from predict import ECGPredictor
from train import loader, output_path, seed_all

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='results/paired_v2/best.pt')
    parser.add_argument('--output', default='results/paired_v2/test')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    folder = output_path(args.output)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Evaluation output must be new or empty')
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    seed_all(args.seed)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    data, stats = load_data('train' if args.validation_only else 'test')
    if args.validation_only:
        split = np.load(Path(args.checkpoint).parent / 'split.npz')
        targets, pool = split['validation_indices'], split['train_indices']
        reference_seed = state['config']['seed'] + 1
    else:
        targets = pool = np.arange(len(data['PPG']))
        reference_seed = args.seed
    dataset = PairedDataset(data, stats, targets, pool, reference_seed)
    np.save(folder / 'reference_indices.npy', dataset.refs)
    np.save(folder / 'target_indices.npy', targets)
    batches = loader(dataset, args.batch_size, args.workers, False)
    model = ECGPredictor(args.checkpoint, 'cuda' if torch.cuda.is_available() else 'cpu')
    shape = (len(dataset), 1250, 1)
    outputs = {key: np.lib.format.open_memmap(folder / f'overall_{key}_data.npy', mode='w+', dtype=np.float32, shape=shape)
               for key in ('fake', 'gt', 'gt_ppg', 'std')}
    baselines = {}
    if not args.validation_only:
        for key, path in (
            ('v0', 'v0/results/rectified_flow_personal/mimic-iv-waveform/samples'),
            ('v1', 'v1/results/rectified_flow_private/mimic-iv-waveform/samples'),
        ):
            baselines[key] = {kind: np.load(ROOT.parent / path / f'overall_{kind}_data.npy', mmap_mode='r')
                              for kind in ('gt', 'gt_ppg', 'fake')}
            if any(array.shape != shape for array in baselines[key].values()):
                raise ValueError(f'{key} baseline shape mismatch')
    totals = {key: np.zeros(3) for key in ('v2', *baselines)}
    subject_ids = np.asarray(data['file_name'])[targets]
    subjects, subject_map = np.unique(subject_ids, return_inverse=True)
    cluster = {key: np.zeros((len(subjects), 3)) for key in totals}
    covered, interval_width, offset = 0, 0., 0
    start = time.monotonic()
    for batch_idx, (ppg, ecg, ppg_ref, ecg_ref) in enumerate(batches):
        point, std = model(ppg, ppg_ref, ecg_ref)
        fake = point.cpu().numpy()
        std = std.cpu().numpy()
        real = ecg.numpy()
        input_ppg = ppg.numpy()
        part = slice(offset, offset + len(real))
        arrays = {'fake': fake, 'gt': real, 'gt_ppg': input_ppg, 'std': std}
        for key, array in arrays.items():
            outputs[key][part] = array
        candidates = {'v2': fake}
        for key, baseline in baselines.items():
            if not np.array_equal(real, baseline['gt'][part]) or not np.array_equal(input_ppg, baseline['gt_ppg'][part]):
                raise ValueError(f'{key} target/PPG data or ordering differs at window {offset}')
            candidates[key] = baseline['fake'][part]
        for key, array in candidates.items():
            totals[key] += paired_sums(real, array)
            error = zscore_numpy(real) - zscore_numpy(array)
            per_window = np.column_stack((np.abs(error).sum(axis=(1, 2)), np.square(error).sum(axis=(1, 2)),
                                          np.full(len(real), error.shape[1] * error.shape[2])))
            np.add.at(cluster[key], subject_map[part], per_window)
        covered += (np.abs(zscore_numpy(real) - fake) <= 1.6448536269514722 * std).sum()
        interval_width += (2 * 1.6448536269514722 * std).sum()
        offset += len(real)
        if batch_idx % 100 == 0:
            print(json.dumps({'evaluated': offset, 'total': len(dataset), 'seconds': time.monotonic() - start}), flush=True)
    if offset != len(dataset):
        raise RuntimeError('Incomplete evaluation')
    for array in outputs.values():
        array.flush()
    results = {key: finish_metrics(*value) for key, value in totals.items()}
    report = {'split': 'fine_tuning_validation' if args.validation_only else 'original_test',
              'samples': offset, 'subjects': len(subjects), 'results': results,
              'checkpoint': str(Path(args.checkpoint).resolve()),
              'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'checkpoint_step': model.training_step, 'checkpoint_selection': 'train.pt validation MAE + RMSE',
              'estimator': 'deterministic paired prediction, not first draw or best-of-K',
              'reference_protocol': 'random different same-subject window in same split (validation uses training pool)',
              'reference_seed': reference_seed, 'original_baseline_reference_indices': 'not saved; identical draws cannot be verified',
              'normalization': 'independent per-window z-score, population std+1e-8; no target alignment or sample filtering',
              'uncertainty': {'type': 'uncalibrated Gaussian pointwise marginal, not joint ECG generation',
                              'coverage90': float(covered / totals['v2'][2]),
                              'mean_width90': float(interval_width / totals['v2'][2])}}
    if baselines:
        report['both_better_than_v0'] = all(results['v2'][key] < results['v0'][key] for key in ('MAE', 'RMSE'))
        report['relative_reduction_vs_v0_percent'] = {key: 100 * (1 - results['v2'][key] / results['v0'][key]) for key in ('MAE', 'RMSE')}
        rng = np.random.default_rng(31415)
        differences = {key: [] for key in ('MAE', 'RMSE')}
        for _ in range(2000):
            sample = rng.integers(0, len(subjects), len(subjects))
            metrics = {key: finish_metrics(*value[sample].sum(axis=0)) for key, value in cluster.items()}
            for key in differences:
                differences[key].append(metrics['v2'][key] - metrics['v0'][key])
        report['patient_cluster_bootstrap95_v2_minus_v0'] = {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in differences.items()}
        np.savez(folder / 'patient_metric_sums.npz', subjects=subjects, **cluster)
    (folder / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
