import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / 'mimic-iv-aligned-ppg_ecgII-processed-filtered'


def split_indices(subjects, seed=42, validation_size=4096):
    if not 0 < validation_size < len(subjects):
        raise ValueError('Invalid validation size')
    remaining = dict(zip(*np.unique(subjects, return_counts=True)))
    validation = []
    for idx in np.random.default_rng(seed).permutation(len(subjects)):
        subject = subjects[idx]
        if remaining[subject] > 2:
            validation.append(idx)
            remaining[subject] -= 1
        if len(validation) == validation_size:
            break
    if len(validation) != validation_size:
        raise ValueError('Insufficient independent reference windows')
    mask = np.ones(len(subjects), dtype=bool)
    mask[validation] = False
    return np.flatnonzero(mask), np.sort(validation)


def subject_groups(subjects, pool):
    pool = np.sort(np.asarray(pool, dtype=np.int64))
    order = pool[np.argsort(subjects[pool], kind='stable')]
    keys, starts = np.unique(subjects[order], return_index=True)
    return dict(zip(keys.tolist(), np.split(order, starts[1:])))


def reference_indices(subjects, targets, pool, rng):
    groups = subject_groups(subjects, pool)
    refs = np.empty(len(targets), dtype=np.int64)
    for pos, idx in enumerate(targets):
        choices = groups[subjects[idx]]
        offset = np.searchsorted(choices, idx)
        present = offset < len(choices) and choices[offset] == idx
        count = len(choices) - int(present)
        if count < 1:
            raise ValueError('Reference must differ from target')
        selected = int(rng.integers(count))
        refs[pos] = choices[selected + int(present and selected >= offset)]
    return refs


def load_data(split):
    if split not in ('train', 'test'):
        raise ValueError('Unknown split')
    data = torch.load(DATA_ROOT / f'{split}.pt', map_location='cpu', weights_only=False, mmap=True)
    if data['PPG'].shape != data['ECG'].shape or data['PPG'].shape[1] != 1250:
        raise ValueError('Unexpected dataset shape')
    with (DATA_ROOT / 'z_score_mean_std.json').open() as handle:
        stats = json.load(handle)['train']
    return data, stats


class PairedDataset(Dataset):
    def __init__(self, data, stats, targets, reference_pool, seed=42, random_references=False):
        self.data, self.stats = data, stats
        self.targets = np.asarray(targets, dtype=np.int64)
        self.subjects = np.asarray(data['file_name'], dtype=np.int64)
        self.groups = subject_groups(self.subjects, reference_pool)
        self.positions = np.full(len(self.subjects), -1, dtype=np.int64)
        for choices in self.groups.values():
            self.positions[choices] = np.arange(len(choices))
        self.random_references = random_references
        self.refs = None if random_references else reference_indices(
            self.subjects, self.targets, reference_pool, np.random.default_rng(seed))
        if any(len(self.groups[self.subjects[idx]]) < 2 and self.positions[idx] >= 0 for idx in self.targets):
            raise ValueError('Every target requires a distinct reference')

    def __len__(self):
        return len(self.targets)

    def normalized(self, key, idx):
        stat = self.stats[key]
        return ((self.data[key][idx].float() - stat['mean']) / (stat['std'] + 1e-8)).unsqueeze(-1)

    def __getitem__(self, item):
        idx = self.targets[item]
        if self.random_references:
            choices = self.groups[self.subjects[idx]]
            offset = self.positions[idx]
            selected = np.random.randint(len(choices) - int(offset >= 0))
            ref = choices[selected + int(offset >= 0 and selected >= offset)]
        else:
            ref = self.refs[item]
        return (self.normalized('PPG', idx), self.normalized('ECG', idx),
                self.normalized('PPG', ref), self.normalized('ECG', ref), int(idx))
