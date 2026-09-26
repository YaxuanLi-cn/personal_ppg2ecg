import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from personal_ppg2ecg.v2.metrics import calculate_fd_for_small_sample, ecg_bpm_array, heart_rate_metrics, zscore_numpy


def _heart_rate_chunk(args):
    samples_dir, start, end, sampling_rate = args
    real = np.load(Path(samples_dir) / 'overall_gt_data.npy', mmap_mode='r')[start:end].astype(np.float64)
    fake = np.load(Path(samples_dir) / 'overall_fake_data.npy', mmap_mode='r')[start:end].astype(np.float64)
    return ecg_bpm_array(real, sampling_rate, 10), ecg_bpm_array(fake, sampling_rate, 10, filter=True)


def compute_heart_rates(samples_dir, sampling_rate=125, workers=8, chunk_size=512):
    if workers < 1 or chunk_size < 1:
        raise ValueError('workers and chunk_size must be positive')
    samples_dir = Path(samples_dir).resolve()
    count = len(np.load(samples_dir / 'overall_gt_data.npy', mmap_mode='r'))
    jobs = [(str(samples_dir), start, min(start + chunk_size, count), sampling_rate)
            for start in range(0, count, chunk_size)]
    real_bpm, fake_bpm = [], []
    processed = 0
    pool = None
    try:
        if workers == 1:
            results = map(_heart_rate_chunk, jobs)
        else:
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
            results = pool.map(_heart_rate_chunk, jobs)
        for real, fake in results:
            real_bpm.append(real)
            fake_bpm.append(fake)
            processed += len(real)
            if processed % (20 * chunk_size) == 0 or processed == count:
                print(f'Heart-rate extraction: {processed}/{count}', flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return np.concatenate(real_bpm), np.concatenate(fake_bpm)


def evaluate(samples_dir, sampling_rate=125, normalize=True, workers=8, seed=42, output_json=None):
    samples_dir = Path(samples_dir).resolve()
    output_json = Path(output_json).resolve() if output_json else samples_dir / 'five_metrics.json'
    if ROOT not in output_json.resolve().parents:
        raise ValueError('Evaluation report must stay inside v2; use --output_json')
    if not output_json.parent.is_dir():
        raise ValueError('Evaluation output parent directory must exist')
    if sampling_rate <= 0 or workers < 1:
        raise ValueError('sampling_rate and workers must be positive')
    np.random.seed(seed)
    gt = np.load(samples_dir / 'overall_gt_data.npy', mmap_mode='r').astype(np.float64)
    fake = np.load(samples_dir / 'overall_fake_data.npy', mmap_mode='r').astype(np.float64)
    if gt.ndim == 2:
        gt = gt[..., None]
    if fake.ndim == 2:
        fake = fake[..., None]
    if gt.shape != fake.shape or gt.ndim != 3 or not len(gt):
        raise ValueError('Expected matching nonempty [N,L,C] arrays')
    if not np.isfinite(gt).all() or not np.isfinite(fake).all():
        raise ValueError('Nonfinite input signals')
    print(f'\n===== {samples_dir} =====', flush=True)
    print(f'GT shape {gt.shape} | Fake shape {fake.shape}', flush=True)
    count = len(gt)
    start = time.monotonic()
    gt_n, fake_n = (zscore_numpy(gt), zscore_numpy(fake)) if normalize else (gt, fake)
    results = {'MAE': float(np.mean(np.abs(gt_n - fake_n))),
               'RMSE': float(np.sqrt(np.mean((gt_n - fake_n) ** 2)))}
    print(f"MAE {results['MAE']:.4f} | RMSE {results['RMSE']:.4f}", flush=True)
    print('Computing FD: full dataset, PCA=64, eps=1e-4, 3 trials...', flush=True)
    fd_mean, fd_std = calculate_fd_for_small_sample(gt_n, fake_n, pca_dim=64, eps=1e-4, n_trials=3)
    results['FD'] = fd_mean
    print(f'FD {fd_mean:.4f} (trial std {fd_std:.6g})', flush=True)
    del gt_n, fake_n, gt, fake
    print('Computing legacy HR metrics: real unfiltered, fake cleaned at 128 Hz; no protocol corrections.', flush=True)
    diagnostics = {}
    try:
        real_bpm, fake_bpm = compute_heart_rates(samples_dir, sampling_rate, workers)
        results.update(heart_rate_metrics(real_bpm, fake_bpm))
        diagnostics = {'real_nonpositive': int(np.sum(real_bpm <= 0)),
                       'fake_nonpositive': int(np.sum(fake_bpm <= 0)),
                       'real_nan': int(np.isnan(real_bpm).sum()), 'fake_nan': int(np.isnan(fake_bpm).sum())}
    except Exception as error:
        results['MAE_hr_paired'] = None
        results['MAE_hr_group'] = None
        diagnostics = {'error': str(error)}
        print(f'[WARN] MAE_hr failed: {error}', flush=True)
    report = {'metrics': results, 'samples_dir': str(samples_dir), 'samples': count,
              'sampling_rate': sampling_rate, 'normalize': normalize, 'seed': seed,
              'fd': {'pca_dim': 64, 'eps': 1e-4, 'n_trials': 3, 'std': fd_std},
              'heart_rate_diagnostics': diagnostics,
              'protocol': 'v0/v1 legacy: fake-only cleaning at 128 Hz; paired excludes NaN only; group separately excludes nonpositive BPM',
              'seconds': time.monotonic() - start}
    output_json.write_text(json.dumps(report, indent=2))
    print('--- Metrics ---', flush=True)
    for key, value in results.items():
        print(f"{key:13s}: {'N/A' if value is None else f'{value:.4f}'}", flush=True)
    print(f'Saved: {output_json}', flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_dir', required=True)
    parser.add_argument('--sampling_rate', type=int, default=125)
    parser.add_argument('--no_normalize', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_json')
    args = parser.parse_args()
    evaluate(args.samples_dir, sampling_rate=args.sampling_rate, normalize=not args.no_normalize,
             workers=args.workers, seed=args.seed, output_json=args.output_json)
