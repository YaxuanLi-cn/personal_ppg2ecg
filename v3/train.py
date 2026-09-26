import argparse
import copy
import json
import time

import numpy as np
import torch

from personal_ppg2ecg.v3.data import load_data, PairedDataset
from personal_ppg2ecg.v3.metrics import finish_metrics, paired_sums, zscore_numpy, calculate_fd_for_small_sample
from personal_ppg2ecg.v3.paired_model import PairedECGHead, distribution_objective
from personal_ppg2ecg.v3.pretrained import FrozenConditions, SHARED_CHECKPOINT, PATIENT_CHECKPOINT
from personal_ppg2ecg.v3.runtime import V2_CHECKPOINT, V2_SPLIT, output_path, seed_all, loader, move


@torch.no_grad()
def validation_cache(conditions, batches, device):
    cached = []
    for batch in batches:
        ppg, ecg, ppg_ref, ecg_ref = move(batch, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            shared, patient = conditions(ppg, ppg_ref, ecg_ref)
        cached.append(tuple(x.cpu() for x in (ppg, ecg, shared, patient)))
    return cached


@torch.no_grad()
def validate(model, cache, device, calculate_fd=False):
    model.eval()
    real, fake = [], []
    for batch in cache:
        ppg, ecg, shared, patient = move(batch, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            predicted, _ = model(ppg, shared, patient)
        real.append(ecg.cpu().numpy())
        fake.append(predicted.float().cpu().numpy())
    real, fake = np.concatenate(real), np.concatenate(fake)
    result = finish_metrics(*paired_sums(real, fake))
    if calculate_fd:
        rng_state = np.random.get_state()
        np.random.seed(42)
        result['FD_one_trial'] = calculate_fd_for_small_sample(zscore_numpy(real), zscore_numpy(fake), n_trials=1)[0]
        np.random.set_state(rng_state)
    return result


def save(path, model, ema, optimizer, scheduler, step, config, validation):
    state = {'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'step': step, 'config': config, 'validation': validation}
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/spectral_v3')
    parser.add_argument('--init-checkpoint', default=str(V2_CHECKPOINT))
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--spectral-weight', type=float, default=2.0)
    parser.add_argument('--stft-weight', type=float, default=0.2)
    parser.add_argument('--eval-every', type=int, default=2000)
    parser.add_argument('--save-every', type=int, default=4000)
    parser.add_argument('--validation-limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.save_every) < 1:
        raise ValueError('Training counts must be positive')
    folder = output_path(args.output)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(folder)
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    torch.set_num_threads(6)
    device = torch.device('cuda')
    initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
    config = dict(vars(args), width=initial['config']['width'], amp=True,
                  shared_checkpoint=str(SHARED_CHECKPOINT), patient_checkpoint=str(PATIENT_CHECKPOINT),
                  data_used='train.pt only', validation_split=str(V2_SPLIT), test_used_for_selection=False)
    (folder / 'config.json').write_text(json.dumps(config, indent=2))
    data, stats = load_data('train')
    split = np.load(V2_SPLIT)
    training, validation = split['train_indices'], split['validation_indices']
    if args.validation_limit:
        validation = validation[:args.validation_limit]
    train_dataset = PairedDataset(data, stats, training, training, args.seed, random_references=True)
    valid_dataset = PairedDataset(data, stats, validation, training, 43)
    conditions = FrozenConditions().to(device).eval()
    model = PairedECGHead(config['width']).to(device)
    model.load_state_dict(initial['ema'], strict=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    del initial
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * 0.1)
    cache = validation_cache(conditions, loader(valid_dataset, args.batch_size, args.workers), device)
    baseline = validate(ema, cache, device, len(validation) >= 64)
    (folder / 'initial_validation.json').write_text(json.dumps(baseline, indent=2))
    print(json.dumps({'initial_validation': baseline, 'config': config}), flush=True)
    batches = loader(train_dataset, args.batch_size, args.workers, True, seed=args.seed + 100)
    iterator = iter(batches)
    best_score, best = float('inf'), None
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            batch = next(iterator)
        ppg, ecg, ppg_ref, ecg_ref = move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            shared, patient = conditions(ppg, ppg_ref, ecg_ref)
            point, scale = model(ppg, shared, patient)
        losses = distribution_objective(point, scale, ecg, args.spectral_weight, args.stft_weight)
        if not torch.isfinite(losses['loss']):
            raise FloatingPointError(f'Nonfinite loss at {step}')
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            decay = min(0.995, (1 + step) / (10 + step))
            for average, parameter in zip(ema.parameters(), model.parameters()):
                average.lerp_(parameter, 1 - decay)
        if step == 1 or step % 100 == 0:
            record = {'step': step, 'seconds_per_step': (time.monotonic() - start) / step,
                      **{key: float(value.detach()) for key, value in losses.items()}}
            print(json.dumps(record), flush=True)
            with (folder / 'train.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
        if step % args.eval_every == 0 or step == args.steps:
            metrics = validate(ema, cache, device, len(validation) >= 64)
            score = metrics['MAE'] + metrics['RMSE'] + 0.02 * metrics.get('FD_one_trial', 0.)
            eligible = metrics['MAE'] < baseline['MAE'] and metrics['RMSE'] < baseline['RMSE']
            if best is None or (eligible and (not best['paired_improved'] or score < best_score)):
                best_score, best = score, dict(metrics, step=step, paired_improved=eligible)
                save(folder / 'best.pt', model, ema, optimizer, scheduler, step, config, best)
            if step % args.save_every == 0 or step == args.steps:
                save(folder / f'checkpoint-{step}.pt', model, ema, optimizer, scheduler, step, config, metrics)
            record = {'step': step, 'validation': metrics, 'best': best}
            with (folder / 'validation.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
    (folder / 'training_complete.json').write_text(json.dumps({'steps': args.steps, 'best': best}, indent=2))


if __name__ == '__main__':
    main()
