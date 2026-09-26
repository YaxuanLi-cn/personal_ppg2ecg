import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from personal_ppg2ecg.v3.metrics import zscore_numpy, paired_sums, finish_metrics
from personal_ppg2ecg.v3.data import reference_indices
from personal_ppg2ecg.v3.paired_model import PairedECGHead, distribution_objective
from personal_ppg2ecg.v3.runtime import output_path


class V3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_metric_parity(self):
        real, fake = np.random.default_rng(42).normal(size=(2, 80, 1250, 1))
        error = zscore_numpy(real) - zscore_numpy(fake)
        got = finish_metrics(*paired_sums(real, fake))
        self.assertAlmostEqual(got['MAE'], np.abs(error).mean(), places=12)
        self.assertAlmostEqual(got['RMSE'], np.sqrt(np.square(error).mean()), places=12)

    def test_no_self_reference_or_validation_reference(self):
        subjects = np.repeat(np.arange(3), 10)
        validation = np.array([0, 10, 20])
        train = np.setdiff1d(np.arange(30), validation)
        for targets in (train, validation):
            refs = reference_indices(subjects, targets, train, np.random.default_rng(42))
            self.assertFalse(np.any(refs == targets))
            self.assertFalse(set(refs) & set(validation))
            np.testing.assert_array_equal(subjects[refs], subjects[targets])

    def test_distribution_loss_gradients_and_checkpoint(self):
        model = PairedECGHead(width=8)
        inputs = (torch.randn(2, 1250, 1), torch.randn(2, 4, 50), torch.randn(2, 256))
        point, scale = model(*inputs)
        losses = distribution_objective(point, scale, torch.randn_like(point))
        losses['loss'].backward()
        self.assertGreater(model.point.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'head.pt'
            torch.save(model.state_dict(), path)
            loaded = PairedECGHead(width=8)
            loaded.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(model(*inputs)[0], loaded(*inputs)[0], atol=0, rtol=0)

    def test_fd_and_legacy_hr_masks(self):
        from personal_ppg2ecg.v3.metrics import calculate_fd_for_small_sample, heart_rate_metrics
        x = np.random.default_rng(7).normal(size=(100, 8))
        fd, _ = calculate_fd_for_small_sample(x, x + 2, pca_dim=None)
        self.assertAlmostEqual(fd, 32, places=9)
        hr = heart_rate_metrics(np.array([60., 80., -1., np.nan, 70.]), np.array([62., 76., -1., 90., np.nan]))
        self.assertEqual(hr, {'MAE_hr_paired': 2., 'MAE_hr_group': 6.})

    def test_transport_fits_only_supplied_training_and_is_batch_independent(self):
        from personal_ppg2ecg.v3.calibration import fit_transport, apply_transport, transport_tensor
        from personal_ppg2ecg.v3.metrics import calculate_fd_for_small_sample
        rng = np.random.default_rng(5)
        real = rng.normal(size=(512, 32, 1))
        fake = real * np.linspace(0.5, 2., 32)[None, :, None]
        layers = fit_transport(real, fake, iterations=2)
        predicted = apply_transport(fake, layers)
        before = calculate_fd_for_small_sample(zscore_numpy(real), zscore_numpy(fake), pca_dim=None)[0]
        after = calculate_fd_for_small_sample(zscore_numpy(real), zscore_numpy(predicted), pca_dim=None)[0]
        self.assertLess(after, before / 10)
        parts = np.concatenate([apply_transport(fake[i:i + 64], layers) for i in range(0, len(fake), 64)])
        np.testing.assert_allclose(predicted, parts, atol=1e-6)
        tensors = [(torch.tensor(x['matrix']), torch.tensor(x['source_mean']), torch.tensor(x['target_mean'])) for x in layers]
        torch_result = transport_tensor(torch.tensor(fake).float(), tensors, 1.).numpy()
        np.testing.assert_allclose(predicted, torch_result, atol=2e-5)
        self.assertEqual(list(inspect.signature(apply_transport).parameters), ['prediction', 'layers', 'strength'])
        for layer in layers:
            self.assertGreaterEqual(np.linalg.eigvalsh(layer['matrix']).min(), 1 / 3 - 1e-5)
            self.assertLessEqual(np.linalg.eigvalsh(layer['matrix']).max(), 3 + 1e-5)

    def test_legacy_metric_functions_match_v2_source(self):
        import ast
        names = {'zscore_numpy', 'calculate_fd_for_small_sample', 'get_Rpeaks_ECG', 'heartbeats_ecg', 'ecg_bpm_array'}
        implementations = []
        for path in (Path(__file__).parent / 'metrics.py', Path(__file__).parent.parent / 'v2/metrics.py'):
            tree = ast.parse(path.read_text())
            implementations.append({node.name: ast.dump(node, include_attributes=False)
                                    for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names})
        self.assertEqual(set(implementations[0]), names)
        self.assertEqual(implementations[0], implementations[1])

    def test_public_inference_excludes_target(self):
        from personal_ppg2ecg.v3.predict import ECGPredictor
        self.assertEqual(list(inspect.signature(ECGPredictor.forward).parameters), ['self', 'ppg', 'ppg_ref', 'ecg_ref'])

    def test_output_isolation(self):
        with self.assertRaises(ValueError):
            output_path('../v2/results/forbidden')
        self.assertEqual(list(inspect.signature(PairedECGHead.forward).parameters), ['self', 'ppg', 'shared', 'patient'])


if __name__ == '__main__':
    unittest.main()
