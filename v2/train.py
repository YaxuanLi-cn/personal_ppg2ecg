import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import PairedDataset, load_data, split_indices
from metrics import finish_metrics, paired_sums, zscore_numpy
from paired_model import PairedECGHead, paired_objective
from pretrained import FrozenConditions, SHARED_CHECKPOINT, PATIENT_CHECKPOINT

ROOT = Path(__file__).resolve().parent


def output_path(path):
    path = Path(path).resolve()
    if ROOT not in path.parents:
        raise ValueError('All outputs must be inside v2')
    return path


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def loader(dataset, batch_size, workers, shuffle, generator=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0, worker_init_fn=seed_worker,
                      generator=generator, drop_last=shuffle)


def move(batch, device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


@torch.no_grad()
def cache_validation(conditions, batches, device, amp):
    cached = []
    for batch in batches:
        ppg, ecg, ppg_ref, ecg_ref = move(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            shared, patient = conditions(ppg, ppg_ref, ecg_ref)
        cached.append(tuple(tensor.cpu() for tensor in (ppg, ecg, shared, patient)))
    return cached


@torch.no_grad()
def validate(model, cached, device, amp):
    model.eval()
    totals = np.zeros(3)
    covered = 0
    width = 0.
    for batch in cached:
        ppg, ecg, shared, patient = move(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            point, scale = model(ppg, shared, patient)
        real = ecg.cpu().numpy()
        fake = point.float().cpu().numpy()
        std = scale.float().cpu().numpy()
        totals += paired_sums(real, fake)
        covered += (np.abs(zscore_numpy(real) - fake) <= 1.6448536269514722 * std).sum()
        width += (2 * 1.6448536269514722 * std).sum()
    return {**finish_metrics(*totals), 'coverage90_marginal': float(covered / totals[2]),
            'width90_marginal': float(width / totals[2])}


def save_checkpoint(path, model, ema, opt, scheduler, step, config, best, stale):
    state = {'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': opt.state_dict(),
             'scheduler': scheduler.state_dict(), 'step': step, 'config': config,
             'best_validation': best, 'stale_evaluations': stale,
             'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
             'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/paired_v2')
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--min-steps', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--validation-size', type=int, default=4096)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.validation_size, args.eval_every, args.log_every) < 1:
        raise ValueError('Training counts must be positive')
    folder = output_path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'config.json').exists() and not args.resume:
        raise FileExistsError('Use a new run directory or --resume')
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda' and not args.no_amp
    config = dict(vars(args), shared_checkpoint=str(SHARED_CHECKPOINT), patient_checkpoint=str(PATIENT_CHECKPOINT),
                  estimator='deterministic point head with separate Gaussian marginal uncertainty',
                  validation_protocol='train.pt fine-tuning holdout; frozen v1 encoders pretrained on all train.pt',
                  test_used_for_selection=False, device=str(device), amp=amp)
    data, stats = load_data('train')
    subjects = np.asarray(data['file_name'], dtype=np.int64)
    training, validation = split_indices(subjects, args.seed, args.validation_size)
    config['train_samples'] = len(training)
    config['validation_index_sha256'] = hashlib.sha256(validation.tobytes()).hexdigest()
    train_dataset = PairedDataset(data, stats, training, training, args.seed, random_references=True)
    valid_dataset = PairedDataset(data, stats, validation, training, args.seed + 1)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = loader(train_dataset, args.batch_size, args.workers, True, generator)
    valid_loader = loader(valid_dataset, args.batch_size, args.workers, False)
    conditions = FrozenConditions().to(device).eval()
    model = PairedECGHead(args.width).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    first_step = 1
    best = {'score': float('inf')}
    stale = 0
    if args.resume:
        state = torch.load(folder / 'last.pt', map_location=device, weights_only=False)
        if state['config']['validation_index_sha256'] != config['validation_index_sha256']:
            raise ValueError('Resume validation split changed')
        for key in ('steps', 'batch_size', 'width', 'lr', 'seed', 'amp'):
            if state['config'][key] != config[key]:
                raise ValueError(f'Resume configuration changed: {key}')
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        first_step = state['step'] + 1
        best, stale = state['best_validation'], state['stale_evaluations']
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in state['cuda_rng']])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
    (folder / 'config.json').write_text(json.dumps(config, indent=2))
    np.savez(folder / 'split.npz', train_indices=training, validation_indices=validation)
    np.save(folder / 'validation_reference_indices.npy', valid_dataset.refs)
    print(json.dumps(config, indent=2), flush=True)
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    start = time.monotonic()
    validation_cache = cache_validation(conditions, valid_loader, device, amp)
    print(f'Validation conditions cached in {time.monotonic() - start:.1f}s', flush=True)
    iterator = iter(train_loader)
    start = time.monotonic()
    step = first_step - 1
    for step in range(first_step, args.steps + 1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        ppg, ecg, ppg_ref, ecg_ref = move(batch, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            shared, patient = conditions(ppg, ppg_ref, ecg_ref)
            point, scale = model(ppg, shared, patient)
        losses = paired_objective(point, scale, ecg)
        if not torch.isfinite(losses['loss']):
            raise FloatingPointError(f'Nonfinite training loss at step {step}')
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        scheduler.step()
        with torch.no_grad():
            decay = min(0.995, (1 + step) / (10 + step))
            for averaged, parameter in zip(ema.parameters(), model.parameters()):
                averaged.lerp_(parameter, 1 - decay)
        if step == first_step or step % args.log_every == 0:
            record = {'step': step, 'seconds_per_step': (time.monotonic() - start) / (step - first_step + 1),
                      'lr': opt.param_groups[0]['lr'], **{key: float(value.detach()) for key, value in losses.items()}}
            print(json.dumps(record), flush=True)
            with (folder / 'train.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
        if step % args.eval_every == 0 or step == args.steps:
            metrics = validate(ema, validation_cache, device, amp)
            score = metrics['MAE'] + metrics['RMSE']
            improved = score < best['score']
            stale = 0 if improved else stale + 1
            if improved:
                best = dict(metrics, score=score, step=step)
                save_checkpoint(folder / 'best.pt', model, ema, opt, scheduler, step, config, best, stale)
            save_checkpoint(folder / 'last.pt', model, ema, opt, scheduler, step, config, best, stale)
            record = {'step': step, 'validation': metrics, 'new_best': improved, 'best': best}
            print(json.dumps(record), flush=True)
            with (folder / 'validation.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            if step >= args.min_steps and stale >= args.patience:
                print('Early stopping on fine-tuning validation only', flush=True)
                break
    summary = {'completed_step': step, 'best_validation': best, 'test_evaluated': False}
    (folder / 'training_complete.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
