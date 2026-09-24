import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent


def output_path(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if ROOT not in path.parents:
        raise ValueError('All outputs must stay inside v4')
    return path


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)
    random.seed(torch.initial_seed() % 2 ** 32)


def loader(dataset, batch_size=64, workers=4, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed), drop_last=shuffle)


def move(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_noise(indices, private_dim, length=250, seed=42):
    indices = np.asarray(indices, dtype=np.int64)
    return torch.from_numpy(np.stack([
        np.random.default_rng(np.random.SeedSequence([seed, int(i)])).standard_normal((private_dim, length)).astype(np.float32)
        for i in indices]))


def source_manifest(verify=False):
    path = ROOT / 'results/source_manifest.json'
    manifest = {}
    for version in ('v0', 'v1', 'v2', 'v3'):
        for file in sorted((ROOT.parent / version).rglob('*')):
            if file.is_file() and file.suffix in ('.py', '.yaml', '.yml', '.sh', '.conf') and 'results' not in file.parts:
                manifest[str(file.relative_to(ROOT.parent))] = sha256(file)
    if verify:
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError('Previous-version source files changed')
    elif not path.exists():
        path.write_text(json.dumps(manifest, indent=2))
    else:
        raise FileExistsError(path)
    return manifest


def save_checkpoint(path, state):
    path = output_path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def append_json(path, record):
    with output_path(path).open('a') as handle:
        handle.write(json.dumps(record, allow_nan=False) + '\n')
