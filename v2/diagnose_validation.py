import argparse
import json
from pathlib import Path

import numpy as np
import torch

from personal_ppg2ecg.v2.data import PairedDataset, load_data
from personal_ppg2ecg.v2.metrics import paired_sums, finish_metrics
from personal_ppg2ecg.v2.predict import ECGPredictor
from personal_ppg2ecg.v2.train import loader, output_path, seed_all


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='results/paired_v2/best.pt')
    parser.add_argument('--output', default='results/paired_v2/validation_diagnostics.json')
    args = parser.parse_args()
    torch.set_num_threads(6)
    seed_all(42)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    split = np.load(Path(args.checkpoint).parent / 'split.npz')
    data, stats = load_data('train')
    dataset = PairedDataset(data, stats, split['validation_indices'], split['train_indices'], state['config']['seed'] + 1)
    model = ECGPredictor(args.checkpoint)
    totals = {key: np.zeros(3) for key in ('full', 'zero_shared', 'zero_patient')}
    max_repeat_difference = 0.
    for batch_idx, (ppg, ecg, ppg_ref, ecg_ref) in enumerate(loader(dataset, 128, 4, False)):
        ppg, ppg_ref, ecg_ref = (tensor.to(model.device) for tensor in (ppg, ppg_ref, ecg_ref))
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=model.amp):
            shared, patient = model.conditions(ppg, ppg_ref, ecg_ref)
            conditions = {'full': (shared, patient), 'zero_shared': (torch.zeros_like(shared), patient),
                          'zero_patient': (shared, torch.zeros_like(patient))}
            for key, values in conditions.items():
                prediction = model.head(ppg, *values)[0]
                totals[key] += paired_sums(ecg.numpy(), prediction.float().cpu().numpy())
            if batch_idx == 0:
                first, _ = model(ppg, ppg_ref, ecg_ref)
                second, _ = model(ppg, ppg_ref, ecg_ref)
                max_repeat_difference = (first - second).abs().max().item()
                if max_repeat_difference != 0:
                    raise AssertionError('Repeated inference is not deterministic')
    report = {'split': 'train.pt fine-tuning validation only', 'samples': len(dataset), 'checkpoint_step': state['step'],
              'results': {key: finish_metrics(*values) for key, values in totals.items()},
              'repeat_prediction_max_abs_difference': max_repeat_difference,
              'note': 'Zero-condition sensitivity is diagnostic, not a separately retrained ablation.'}
    output_path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
