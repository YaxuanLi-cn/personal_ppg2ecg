import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

from personal_ppg2ecg.v4.data import PairedDataset, load_data, reference_indices, split_indices, subject_groups
from personal_ppg2ecg.v4.metrics import (paired_sums, finish_metrics, zscore_numpy, zscore_tensor,
                     calculate_fd_for_small_sample, ecg_bpm_array, heart_rate_metrics)
from personal_ppg2ecg.v4.model import SharedPrivateModel
from personal_ppg2ecg.v4.pretrained import FrozenConditions
from personal_ppg2ecg.v4.predict import ECGPredictor
from personal_ppg2ecg.v4.runtime import fixed_noise, save_checkpoint

V2_ROOT = ROOT.parent / 'v2'


def _load_v2_metrics():
    spec = importlib.util.spec_from_file_location('v2_metrics_parity', V2_ROOT / 'metrics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensors(batch=4, seed=0):
    rng = np.random.default_rng(seed)
    make = lambda: torch.from_numpy(rng.standard_normal((batch, 1250, 1)).astype(np.float32))
    return make(), make(), make(), make()


class TestShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SharedPrivateModel(width=32, private_dim=8)
        cls.ppg, cls.ecg, cls.ppg_ref, cls.ecg_ref = _tensors()
        cls.anchor = torch.randn(4, 4, 50)
        cls.patient = torch.randn(4, 256)

    def test_shared_and_private_shapes(self):
        shared = self.model.shared_encoder(self.ppg, self.anchor)
        private = self.model.private_encoder(self.ecg)
        self.assertEqual(tuple(shared.shape), (4, 32, 250))
        self.assertEqual(tuple(private.shape), (4, 8, 250))

    def test_predictor_and_decoder_shapes(self):
        shared = self.model.shared_encoder(self.ppg, self.anchor)
        private = self.model.private_predictor(shared, self.patient)
        self.assertEqual(tuple(private.shape), (4, 8, 250))
        self.assertEqual(tuple(self.model.decoder(shared, private).shape), (4, 1250, 1))
        with self.assertRaises(ValueError):
            self.model.decoder(shared, torch.randn(4, 9, 250))

    def test_flow_and_sampling_shapes(self):
        shared = self.model.shared_encoder(self.ppg, self.anchor)
        state = torch.randn(4, 8, 250)
        velocity = self.model.flow(state, torch.rand(4), shared, self.patient)
        self.assertEqual(tuple(velocity.shape), (4, 8, 250))
        noise = torch.randn(4, 8, 250)
        private = self.model.sample_private(shared, self.patient, noise, steps=5)
        self.assertEqual(tuple(private.shape), (4, 8, 250))
        output = self.model(self.ppg, self.anchor, self.patient, noise, steps=5)
        self.assertEqual(tuple(output.shape), (4, 1250, 1))
        self.assertTrue(torch.isfinite(output).all())
        with self.assertRaises(ValueError):
            self.model.sample_private(shared, self.patient, torch.randn(4, 8, 251))

    def test_inference_signature_has_no_target_ecg(self):
        forward_params = inspect.signature(self.model.forward).parameters
        self.assertNotIn('ecg', forward_params)
        call_params = inspect.signature(ECGPredictor.__call__).parameters
        self.assertNotIn('ecg', set(call_params) - {'ecg_ref'})

    def test_private_path_changes_output(self):
        self.model.eval()
        with torch.no_grad():
            shared = self.model.shared_encoder(self.ppg, self.anchor)
            noise = torch.randn(4, 8, 250)
            out_a = self.model(self.ppg, self.anchor, self.patient, noise, steps=3)
            out_b = self.model(self.ppg, self.anchor, self.patient * 2, noise, steps=3)
            self.assertFalse(torch.equal(out_a, out_b))
            noise2 = torch.randn(4, 8, 250)
            out_c = self.model(self.ppg, self.anchor, self.patient, noise2, steps=3)
            self.assertFalse(torch.equal(out_a, out_c))


class TestGradients(unittest.TestCase):
    def test_representation_gradients(self):
        model = SharedPrivateModel(width=32, private_dim=8)
        ppg, ecg, _, _ = _tensors(8)
        anchor_p, anchor_e = torch.randn(8, 4, 50), torch.randn(8, 4, 50)
        patient = torch.randn(8, 256)
        terms = model.representation_loss(ppg, ecg, anchor_p, anchor_e, patient)
        terms['loss'].backward()
        for name in ('shared_encoder', 'private_encoder', 'private_predictor', 'decoder', 'ppg_decoder'):
            grads = [p.grad for p in getattr(model, name).parameters()]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads), name)
        self.assertTrue(all(p.grad is None for p in model.flow.parameters()))

    def test_flow_gradients_isolated(self):
        model = SharedPrivateModel(width=32, private_dim=8)
        model.freeze_representation()
        ppg, ecg, _, _ = _tensors(8)
        terms = model.flow_loss(ppg, ecg, torch.randn(8, 4, 50), torch.randn(8, 256))
        terms['loss'].backward()
        grads = [p.grad for p in model.flow.parameters()]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))
        self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))


class TestCheckpoint(unittest.TestCase):
    def test_roundtrip(self):
        model = SharedPrivateModel(width=32, private_dim=8)
        with tempfile.TemporaryDirectory(dir=ROOT / 'results') as tmp:
            path = Path(tmp) / 'ck.pt'
            save_checkpoint(path, {'model': model.state_dict(),
                                   'ema': {n: p.detach().cpu() for n, p in model.named_parameters()},
                                   'config': {'model': model.configuration()},
                                   'selection': {'test_used': False}, 'step': 3})
            state = torch.load(path, map_location='cpu', weights_only=False)
        clone = SharedPrivateModel(**state['config']['model'])
        clone.load_state_dict(state['ema'])
        ppg, *_ = _tensors()
        anchor, patient, noise = torch.randn(4, 4, 50), torch.randn(4, 256), torch.randn(4, 8, 250)
        model.eval(); clone.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(ppg, anchor, patient, noise, 4),
                                        clone(ppg, anchor, patient, noise, 4)))


class TestDataProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.stats = load_data('train')
        cls.subjects = np.asarray(cls.data['file_name'], dtype=np.int64)

    def test_split_reproduces_v2(self):
        fit, val = split_indices(self.subjects, seed=42, validation_size=4096)
        split = np.load(V2_ROOT / 'results/paired_v2/split.npz')
        self.assertTrue(np.array_equal(fit, split['train_indices']))
        self.assertTrue(np.array_equal(val, split['validation_indices']))

    def test_references_same_subject_distinct_deterministic(self):
        targets = np.arange(2000)
        pool = np.arange(len(self.subjects))
        rng_a = np.random.default_rng(7)
        refs_a = reference_indices(self.subjects, targets, pool, rng_a)
        refs_b = reference_indices(self.subjects, targets, pool, np.random.default_rng(7))
        self.assertTrue(np.array_equal(refs_a, refs_b))
        self.assertTrue((self.subjects[refs_a] == self.subjects[targets]).all())
        self.assertFalse((refs_a == targets).any())

    def test_validation_pool_excludes_validation(self):
        split = np.load(V2_ROOT / 'results/paired_v2/split.npz')
        fit, val = split['train_indices'], split['validation_indices']
        dataset = PairedDataset(self.data, self.stats, val[:512], fit, seed=43)
        self.assertTrue(np.isin(dataset.refs, fit).all())
        self.assertEqual(np.intersect1d(dataset.refs, val).size, 0)
        groups = subject_groups(self.subjects, fit)
        for t, r in zip(val[:512], dataset.refs):
            self.assertEqual(self.subjects[t], self.subjects[r])

    def test_test_reference_draws_match_v2(self):
        test_data, _ = load_data('test')
        subjects = np.asarray(test_data['file_name'], dtype=np.int64)
        targets = np.arange(20000)
        refs = reference_indices(subjects, targets, np.arange(len(subjects)), np.random.default_rng(42))
        saved = np.load(V2_ROOT / 'results/paired_v2/test/reference_indices.npy')
        self.assertTrue(np.array_equal(refs, saved[:20000]))


class TestMetricParity(unittest.TestCase):
    def test_waveform_metric_parity_with_v2(self):
        v2m = _load_v2_metrics()
        rng = np.random.default_rng(1)
        real = rng.standard_normal((64, 1250, 1)).astype(np.float64)
        fake = real + rng.standard_normal((64, 1250, 1)).astype(np.float64) * 0.3
        self.assertEqual(paired_sums(real, fake), v2m.paired_sums(real, fake))
        self.assertEqual(finish_metrics(*paired_sums(real, fake)),
                         v2m.finish_metrics(*v2m.paired_sums(real, fake)))

    def test_fd_parity_with_v2(self):
        v2m = _load_v2_metrics()
        rng = np.random.default_rng(2)
        gt = rng.standard_normal((256, 1250, 1)).astype(np.float64)
        fake = gt + rng.standard_normal((256, 1250, 1)).astype(np.float64) * 0.5
        np.random.seed(42)
        mine = calculate_fd_for_small_sample(zscore_numpy(gt), zscore_numpy(fake), n_trials=1)
        np.random.seed(42)
        theirs = v2m.calculate_fd_for_small_sample(v2m.zscore_numpy(gt), v2m.zscore_numpy(fake), n_trials=1)
        self.assertAlmostEqual(mine[0], theirs[0], places=6)

    def test_hr_metric_structure(self):
        metrics = heart_rate_metrics(np.array([60., 70., np.nan]), np.array([62., -1., 80.]))
        self.assertEqual(set(metrics), {'MAE_hr_paired', 'MAE_hr_group'})
        self.assertAlmostEqual(metrics['MAE_hr_paired'], np.mean([2., 71.]))
        self.assertAlmostEqual(metrics['MAE_hr_group'], abs(np.mean([60., 70.]) - np.mean([62., 80.])))


class TestFrozenConditions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conditions = FrozenConditions()

    def test_anchor_and_patient_shapes(self):
        ppg, _, ppg_ref, ecg_ref = _tensors(2)
        with torch.no_grad():
            anchor = self.conditions.shared(ppg)
            patient = self.conditions.patient(ppg_ref, ecg_ref)
        self.assertEqual(tuple(anchor.shape), (2, 4, 50))
        self.assertEqual(tuple(patient.shape), (2, 256))
        self.assertTrue(torch.isfinite(anchor).all() and torch.isfinite(patient).all())
        norms = patient.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_deterministic_and_frozen(self):
        self.assertFalse(any(p.requires_grad for p in self.conditions.parameters()))
        ppg, _, ppg_ref, ecg_ref = _tensors(2)
        with torch.no_grad():
            a1, h1 = self.conditions.shared(ppg), self.conditions.patient(ppg_ref, ecg_ref)
            a2, h2 = self.conditions.shared(ppg), self.conditions.patient(ppg_ref, ecg_ref)
        self.assertTrue(torch.equal(a1, a2) and torch.equal(h1, h2))
        _, _, other_ref, other_ecg = _tensors(2, seed=9)
        with torch.no_grad():
            h3 = self.conditions.patient(other_ref, other_ecg)
        self.assertFalse(torch.equal(h1, h3))


class TestPredictorInterface(unittest.TestCase):
    def test_predictor_modes_and_determinism(self):
        model = SharedPrivateModel(width=32, private_dim=8)
        with tempfile.TemporaryDirectory(dir=ROOT / 'results') as tmp:
            path = Path(tmp) / 'inference.pt'
            torch.save({'ema': {n: p.detach().cpu() for n, p in model.named_parameters()},
                        'config': {'model': model.configuration()}, 'step': 1,
                        'selection': {'test_used': False}}, path)
            for mode in ('mean', 'sample'):
                predictor = ECGPredictor(path, device='cpu', mode=mode, flow_steps=3)
                ppg, _, ppg_ref, ecg_ref = _tensors(2)
                idx = np.array([10, 20])
                out1 = predictor(ppg, ppg_ref, ecg_ref, indices=idx)
                out2 = predictor(ppg, ppg_ref, ecg_ref, indices=idx)
                self.assertEqual(tuple(out1.shape), (2, 1250, 1))
                self.assertTrue(torch.isfinite(out1).all())
                self.assertTrue(torch.equal(out1, out2))
                out3 = predictor(ppg, ppg_ref, ecg_ref, indices=np.array([11, 21]))
                if mode == 'sample':
                    self.assertFalse(torch.equal(out1, out3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
