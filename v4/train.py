import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from personal_ppg2ecg.v4.data import PairedDataset, load_data
from personal_ppg2ecg.v4.metrics import paired_sums, finish_metrics
from personal_ppg2ecg.v4.model import SharedPrivateModel
from personal_ppg2ecg.v4.pretrained import FrozenConditions
from personal_ppg2ecg.v4.runtime import ROOT, output_path, seed_all, loader, move, fixed_noise, save_checkpoint, append_json

V2_SPLIT = ROOT.parent / 'v2/results/paired_v2/split.npz'


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = None

    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def store(self, model):
        self.backup = {n: p.detach().clone() for n, p in model.named_parameters() if n in self.shadow}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    p.copy_(self.shadow[n])

    def restore(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    p.copy_(self.backup[n])
        self.backup = None

    def state_dict(self):
        return {n: t.clone() for n, t in self.shadow.items()}


def lr_at(step, total, base, warmup=500, floor=0.05):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def validate(model, conditions, dataset, device, mode, flow_steps=8, batch_size=512, workers=4, seed=42):
    model.eval()
    absolute = squared = count = 0
    for ppg, ecg, ppg_ref, ecg_ref, idx in loader(dataset, batch_size, workers, False):
        ppg, ecg, ppg_ref, ecg_ref, idx = move((ppg, ecg, ppg_ref, ecg_ref, idx), device)
        anchor, patient = conditions.shared(ppg), conditions.patient(ppg_ref, ecg_ref)
        if mode == 'mean':
            shared = model.shared_encoder(ppg, anchor)
            private = model.private_predictor(shared, patient)
            fake = model.decoder(shared, private)
        else:
            noise = fixed_noise(idx.cpu().numpy(), model.private_dim, seed=seed).to(device)
            fake = model(ppg, anchor, patient, noise, flow_steps)
        a, s, n = paired_sums(ecg.cpu().numpy(), fake.float().cpu().numpy())
        absolute += a
        squared += s
        count += n
    return finish_metrics(absolute, squared, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('rep', 'flow'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--init-checkpoint')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-every', type=int, default=2000)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--flow-steps', type=int, default=8)
    parser.add_argument('--spectral-weight', type=float, default=1.0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    folder = output_path(args.output)
    if folder.exists() and any(folder.iterdir()) and not args.resume:
        raise FileExistsError('Training output must be new, or pass --resume')
    folder.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_all(args.seed)
    torch.set_num_threads(6)

    data, stats = load_data('train')
    split = np.load(V2_SPLIT)
    training, validation = split['train_indices'], split['validation_indices']
    fit_dataset = PairedDataset(data, stats, training, training, args.seed, random_references=True)
    val_dataset = PairedDataset(data, stats, validation, training, args.seed + 1)
    np.savez(folder / 'split.npz', train_indices=training, validation_indices=validation,
             validation_reference_indices=val_dataset.refs)

    model = SharedPrivateModel().to(device)
    start_step = 0
    best = math.inf
    config = {'stage': args.stage, 'model': model.configuration(), 'seed': args.seed,
              'batch_size': args.batch_size, 'lr': args.lr, 'steps': args.steps,
              'flow_steps': args.flow_steps, 'spectral_weight': args.spectral_weight,
              'val_size': len(validation), 'data_used': 'train.pt only',
              'validation_split': str(V2_SPLIT), 'test_used_for_selection': False}

    conditions = FrozenConditions().to(device)
    if args.stage == 'flow':
        if not args.init_checkpoint:
            raise ValueError('--stage flow requires --init-checkpoint')
        state = torch.load(output_path(args.init_checkpoint) if not Path(args.init_checkpoint).is_absolute()
                           else args.init_checkpoint, map_location='cpu', weights_only=False)
        if state.get('selection', {}).get('test_used') is not False:
            raise ValueError('Stage-rep checkpoint must be selected without test data')
        model.load_state_dict(state['ema'], strict=True)
        model.freeze_representation()
        trainable = [p for p in model.flow.parameters()]
        config['init_checkpoint'] = str(Path(args.init_checkpoint).resolve())
    else:
        trainable = [p for n, p in model.named_parameters() if not n.startswith('flow.')]

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    ema = EMA(model, args.ema_decay)

    if args.resume:
        state = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        for n in list(ema.shadow):
            if n in state['ema']:
                ema.shadow[n] = state['ema'][n].to(device)
        start_step, best = state['step'], state['best']

    batches = loader(fit_dataset, args.batch_size, args.workers, True, args.seed)
    iterator = iter(batches)
    history = folder / 'metrics.jsonl'
    start = time.monotonic()
    model.train() if args.stage == 'rep' else model.flow.train()

    for step in range(start_step, args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            batch = next(iterator)
        ppg, ecg, ppg_ref, ecg_ref, _ = move(batch, device)
        with torch.no_grad():
            anchor_p = conditions.shared(ppg)
            anchor_e = conditions.shared(ecg)
            patient = conditions.patient(ppg_ref, ecg_ref)
        if args.stage == 'rep':
            terms = model.representation_loss(ppg, ecg, anchor_p, anchor_e, patient, args.spectral_weight)
        else:
            terms = model.flow_loss(ppg, ecg, anchor_p, patient)
        loss = terms['loss']
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step, args.steps, args.lr)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        ema.update(model)

        if step % 100 == 0:
            record = {'step': step, 'lr': optimizer.param_groups[0]['lr'],
                      'seconds': time.monotonic() - start,
                      **{k: float(v.detach()) for k, v in terms.items()}}
            append_json(history, record)
            print(json.dumps(record), flush=True)

        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            ema.store(model)
            mode = 'mean' if args.stage == 'rep' else 'sample'
            metrics = validate(model, conditions, val_dataset, device, mode, args.flow_steps, seed=args.seed + 1)
            score = metrics['MAE'] + metrics['RMSE']
            record = {'step': step + 1, 'mode': mode, **metrics, 'score': score, 'best': score < best}
            append_json(history, {'validation': record})
            print(json.dumps({'validation': record}), flush=True)
            full_ema = {n: (ema.shadow[n].cpu() if n in ema.shadow else p.detach().cpu())
                        for n, p in model.named_parameters()}
            state = {'model': model.state_dict(), 'ema': full_ema,
                     'optimizer': optimizer.state_dict(), 'step': step + 1, 'best': min(best, score),
                     'config': config, 'selection': {'objective': 'train.pt validation MAE+RMSE',
                     'mode': mode, 'flow_steps': args.flow_steps, 'test_used': False,
                     'validation_indices': 'v2/results/paired_v2/split.npz'},
                     'metrics': metrics}
            save_checkpoint(folder / 'last.pt', state)
            if score < best:
                best = score
                save_checkpoint(folder / 'best.pt', state)
            ema.restore(model)
            model.train() if args.stage == 'rep' else model.flow.train()

    print(json.dumps({'complete': True, 'best_score': best, 'seconds': time.monotonic() - start}), flush=True)


if __name__ == '__main__':
    main()
