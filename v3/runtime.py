import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
V2_CHECKPOINT = ROOT.parent / 'v2/results/paired_v2/best.pt'
V2_SPLIT = ROOT.parent / 'v2/results/paired_v2/split.npz'


def output_path(path):
    path = Path(path).resolve()
    if ROOT not in path.parents:
        raise ValueError('All outputs must stay inside v3')
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


def loader(dataset, batch_size=128, workers=4, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), drop_last=shuffle)


def move(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


def source_manifest(verify=False):
    path = ROOT / 'results/source_manifest.json'
    manifest = {}
    for version in ('v0', 'v1', 'v2'):
        for file in sorted((ROOT.parent / version).rglob('*')):
            if file.is_file() and file.suffix in ('.py', '.yaml', '.yml', '.sh', '.conf') and 'results' not in file.parts:
                manifest[str(file.relative_to(ROOT.parent))] = hashlib.sha256(file.read_bytes()).hexdigest()
    if verify:
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError('Previous-version sources changed')
    elif path.exists():
        raise FileExistsError(path)
    else:
        path.write_text(json.dumps(manifest, indent=2))
    print(f'Verified/recorded {len(manifest)} previous-version source hashes', flush=True)
