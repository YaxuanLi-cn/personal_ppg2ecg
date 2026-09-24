import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data import load_data, PairedDataset
from predict import ECGPredictor
from runtime import V2_SPLIT, V2_CHECKPOINT, loader, output_path, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=str(V2_CHECKPOINT))
    parser.add_argument('--output', required=True)
    parser.add_argument('--calibration-size', type=int, default=65536)
    args = parser.parse_args()
    folder = output_path(args.output)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    seed_all(42)
    data, stats = load_data('train')
    split = np.load(V2_SPLIT)
    training, validation = split['train_indices'], split['validation_indices']
    calibration = np.sort(np.random.default_rng(314).choice(training, min(args.calibration_size, len(training)), replace=False))
    if np.intersect1d(calibration, validation).size:
        raise AssertionError('Calibration and validation overlap')
    model = ECGPredictor(args.checkpoint)
    for name, targets, seed in (('calibration', calibration, 73), ('validation', validation, 43)):
        dataset = PairedDataset(data, stats, targets, training, seed)
        shape = (len(targets), 1250, 1)
        real = np.lib.format.open_memmap(folder / f'{name}_real.npy', mode='w+', dtype=np.float32, shape=shape)
        fake = np.lib.format.open_memmap(folder / f'{name}_fake.npy', mode='w+', dtype=np.float32, shape=shape)
        np.save(folder / f'{name}_indices.npy', targets)
        np.save(folder / f'{name}_references.npy', dataset.refs)
        offset = 0
        for ppg, ecg, ppg_ref, ecg_ref in loader(dataset, batch_size=256):
            predicted = model(ppg, ppg_ref, ecg_ref).cpu().numpy()
            real[offset:offset + len(ecg)] = ecg.numpy()
            fake[offset:offset + len(ecg)] = predicted
            offset += len(ecg)
        real.flush()
        fake.flush()
        print(f'Collected train.pt {name}: {offset} windows', flush=True)
    record = {'data_source': 'train.pt only', 'checkpoint': str(Path(args.checkpoint).resolve()),
              'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'calibration_count': len(calibration), 'validation_count': len(validation), 'overlap': 0}
    (folder / 'provenance.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
