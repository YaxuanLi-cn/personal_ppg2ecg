import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from personal_ppg2ecg.v3.calibration import apply_transport, fit_transport
from personal_ppg2ecg.v3.evaluation.run_eval import evaluate
from personal_ppg2ecg.v3.runtime import output_path, V2_SPLIT

TARGETS = {'MAE': 0.39591144402467815, 'RMSE': 0.6698436880509359, 'FD': 0.4554,
           'MAE_hr_paired': 2.2012139397624995, 'MAE_hr_group': 0.5570}


def selection_key(metrics, baseline):
    if any(metrics.get(key) is None or not np.isfinite(metrics[key]) for key in TARGETS):
        return (6, float('inf'), float('inf'))
    regressions = sum(metrics[key] >= baseline[key] for key in TARGETS)
    ratios = np.array([metrics[key] / TARGETS[key] for key in TARGETS])
    return int(regressions), float(ratios.max()), float(np.log(np.maximum(ratios, 1e-8)).mean())


def validate_provenance(cache):
    provenance = json.loads((cache / 'provenance.json').read_text())
    if provenance['data_source'] != 'train.pt only':
        raise ValueError('Calibration requires train.pt data')
    split = np.load(V2_SPLIT)
    calibration = np.load(cache / 'calibration_indices.npy')
    validation = np.load(cache / 'validation_indices.npy')
    if not np.array_equal(validation, split['validation_indices']):
        raise ValueError('Validation split changed')
    if np.setdiff1d(calibration, split['train_indices']).size or np.intersect1d(calibration, validation).size:
        raise ValueError('Calibration includes held-out targets')
    checkpoint = Path(provenance['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != provenance['checkpoint_sha256']:
        raise ValueError('Source checkpoint changed after collecting calibration data')
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline-json')
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--strengths', type=float, nargs='+', default=[0., 0.5, 0.75, 1.])
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--regularization', type=float, default=0.01)
    args = parser.parse_args()
    cache, folder = output_path(args.cache), output_path(args.output)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(folder)
    provenance = validate_provenance(cache)
    folder.mkdir(parents=True, exist_ok=True)
    real = np.load(cache / 'validation_real.npy', mmap_mode='r')
    fake = np.load(cache / 'validation_fake.npy', mmap_mode='r')
    if args.baseline_only:
        np.save(folder / 'overall_gt_data.npy', real)
        np.save(folder / 'overall_fake_data.npy', fake)
        evaluate(folder, workers=args.workers)
        return
    if not args.baseline_json:
        raise ValueError('--baseline-json is required for model selection')
    baseline = json.loads(Path(args.baseline_json).read_text())['metrics']
    real_train = np.load(cache / 'calibration_real.npy', mmap_mode='r')
    fake_train = np.load(cache / 'calibration_fake.npy', mmap_mode='r')
    print(f'Fitting regularized covariance transport on {len(real_train)} training windows; validation excluded', flush=True)
    layers = fit_transport(real_train, fake_train, iterations=args.iterations, regularization=args.regularization)
    torch.save({'layers': layers, 'provenance': provenance, 'regularization': args.regularization}, folder / 'transport.pt')
    records = []
    for strength in args.strengths:
        candidate = folder / f'strength_{strength:g}'
        candidate.mkdir()
        prediction = apply_transport(fake, layers, strength)
        np.save(candidate / 'overall_gt_data.npy', real)
        np.save(candidate / 'overall_fake_data.npy', prediction)
        metrics = evaluate(candidate, workers=args.workers)
        record = {'strength': strength, 'validation': metrics, 'selection_key': selection_key(metrics, baseline)}
        records.append(record)
        (folder / 'candidates.json').write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
    best = min(records, key=lambda item: item['selection_key'])
    if best['selection_key'][0] == 6:
        raise RuntimeError('No candidate has five finite validation metrics')
    selected = dict(best, provenance=provenance, test_used=False,
                    rule='minimize regressions against v2 validation, then worst historical-target-scaled ratio, then mean log ratio',
                    baseline_validation=baseline)
    state = torch.load(provenance['checkpoint'], map_location='cpu', weights_only=False)
    state['transport'] = layers
    state['transport_strength'] = best['strength']
    state['selection'] = selected
    torch.save(state, folder / 'inference.pt')
    (folder / 'selection.json').write_text(json.dumps(selected, indent=2))
    print(json.dumps({'selected': selected}, indent=2), flush=True)


if __name__ == '__main__':
    main()
