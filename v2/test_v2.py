import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from metrics import finish_metrics, paired_sums, zscore_numpy, zscore_tensor
from data import PairedDataset, split_indices, reference_indices
from train import output_path
from predict import ECGPredictor
from paired_model import PairedECGHead, paired_objective


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        torch.backends.cudnn.deterministic = True

    def test_metrics_match_original(self):
        rng = np.random.default_rng(42)
        gt = rng.normal(size=(17, 1250, 1))
        fake = rng.normal(size=gt.shape)
        diff = zscore_numpy(gt) - zscore_numpy(fake)
        totals = np.zeros(3)
        for start in range(0, len(gt), 6):
            totals += paired_sums(gt[start:start + 6], fake[start:start + 6])
        result = finish_metrics(*totals)
        json.dumps(dict(result, improved=result['MAE'] < 2))
        self.assertAlmostEqual(result['MAE'], np.abs(diff).mean(), places=12)
        self.assertAlmostEqual(result['RMSE'], np.sqrt(np.square(diff).mean()), places=12)
        self.assertLess(np.max(np.abs(zscore_tensor(torch.tensor(gt)).numpy() - zscore_numpy(gt))), 2e-6)

    def test_metrics_constant_nonfinite_and_shape(self):
        x = np.ones((3, 25, 1))
        self.assertEqual(finish_metrics(*paired_sums(x, x)), {'MAE': 0., 'RMSE': 0.})
        with self.assertRaises(ValueError):
            paired_sums(x, np.full_like(x, np.nan))
        with self.assertRaises(ValueError):
            paired_sums(x, x[:2])

    def test_split_and_reference_exclusion(self):
        subjects = np.repeat(np.arange(4), 100)
        train, valid = split_indices(subjects, seed=42, validation_size=40)
        self.assertEqual(len(valid), 40)
        self.assertFalse(set(train) & set(valid))
        self.assertEqual(len(train) + len(valid), len(subjects))
        np.testing.assert_array_equal(valid, split_indices(subjects, 42, 40)[1])
        rng = np.random.default_rng(1)
        for targets, pool in ((train, train), (valid, train)):
            refs = reference_indices(subjects, targets, pool, rng)
            self.assertFalse(np.any(refs == targets))
            self.assertTrue(set(refs).issubset(set(train)))
            np.testing.assert_array_equal(subjects[refs], subjects[targets])

    def test_small_subjects_stay_train(self):
        subjects = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        train, valid = split_indices(subjects, 4, 2)
        self.assertTrue({0, 1}.issubset(set(train)))
        self.assertFalse(set(train) & set(valid))
        with self.assertRaises(ValueError):
            reference_indices(np.array([0, 1]), np.array([0]), np.array([0, 1]), np.random.default_rng(1))

    def test_dataset_reference_never_uses_current_target(self):
        data = {'PPG': torch.arange(20).float()[:, None].repeat(1, 1250),
                'ECG': torch.arange(20).float()[:, None].repeat(1, 1250),
                'file_name': torch.arange(20) // 5}
        stats = {key: {'mean': 0., 'std': 1.} for key in ('PPG', 'ECG')}
        for random in (True, False):
            ds = PairedDataset(data, stats, np.arange(20), np.arange(19, -1, -1), random_references=random)
            for _ in range(4):
                for idx in range(20):
                    ppg, ecg, ppg_ref, ecg_ref = ds[idx]
                    self.assertFalse(torch.equal(ecg, ecg_ref))
                    self.assertEqual(int(ecg_ref[0, 0]) // 5, idx // 5)
        refs = reference_indices(data['file_name'].numpy(), np.arange(20), np.arange(19, -1, -1), np.random.default_rng(1))
        self.assertFalse(np.any(refs == np.arange(20)))

    def test_output_isolation_and_public_inference_api(self):
        with self.assertRaises(ValueError):
            output_path(Path(__file__).parent.parent / 'v1/results/forbidden')
        self.assertEqual(list(inspect.signature(ECGPredictor.forward).parameters), ['self', 'ppg', 'ppg_ref', 'ecg_ref'])

    def inputs(self):
        return torch.randn(2, 1250, 1), torch.randn(2, 4, 50), torch.randn(2, 256)

    def test_point_head_gradients_and_uncertainty_separation(self):
        model = PairedECGHead(width=8)
        point, scale = model(*self.inputs())
        self.assertEqual(point.shape, (2, 1250, 1))
        self.assertEqual(scale.shape, point.shape)
        self.assertTrue(torch.isfinite(point).all())
        self.assertTrue((scale > 0).all())
        target = torch.randn_like(point)
        losses = paired_objective(point, scale, target)
        losses['loss'].backward()
        self.assertGreater(model.point.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.uncertainty[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        model.zero_grad(set_to_none=True)
        point, scale = model(*self.inputs())
        paired_objective(point, scale, target)['nll'].backward()
        self.assertIsNone(model.point.weight.grad)
        self.assertIsNone(model.stem.weight.grad)

    def test_deterministic_checkpoint_and_no_target_argument(self):
        model = PairedECGHead(width=8).eval()
        inputs = self.inputs()
        with torch.no_grad():
            expected = model(*inputs)[0]
            torch.testing.assert_close(expected, model(*inputs)[0], rtol=0, atol=0)
        self.assertEqual(list(inspect.signature(model.forward).parameters), ['ppg', 'shared', 'patient'])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'model.pt'
            torch.save(model.state_dict(), path)
            restored = PairedECGHead(width=8).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(expected, restored(*inputs)[0], rtol=0, atol=0)


class FiveMetricTests(unittest.TestCase):
    def test_fd_identical_and_translation(self):
        from metrics import calculate_fd_for_small_sample
        rng = np.random.default_rng(19)
        gt = rng.normal(size=(100, 8, 1))
        fd, std = calculate_fd_for_small_sample(gt, gt.copy(), pca_dim=None, n_trials=3)
        self.assertAlmostEqual(fd, 0., places=10)
        self.assertAlmostEqual(std, 0., places=10)
        fd, _ = calculate_fd_for_small_sample(gt, gt + 2., pca_dim=None)
        self.assertAlmostEqual(fd, 32., places=10)

    def test_hr_masks_match_legacy(self):
        from metrics import heart_rate_metrics
        real = np.array([60., 80., -1., np.nan, 70.])
        fake = np.array([62., 76., -1., 90., np.nan])
        result = heart_rate_metrics(real, fake)
        self.assertEqual(result['MAE_hr_paired'], 2.)
        self.assertEqual(result['MAE_hr_group'], 6.)

    def test_legacy_cleaning_rate_is_preserved(self):
        from unittest.mock import patch
        from metrics import ecg_bpm_array
        x = np.zeros((2, 1250, 1))
        with patch('neurokit2.ecg_clean', side_effect=lambda x, **kwargs: x) as clean:
            with patch('metrics.heartbeats_ecg', return_value=([0, 1], [60., 62.])) as beats:
                np.testing.assert_array_equal(ecg_bpm_array(x, 125, 10, filter=True), [61., 61.])
                self.assertEqual(clean.call_args.kwargs['sampling_rate'], 128)
                self.assertEqual(beats.call_args.args[1], 125)
                clean.reset_mock()
                ecg_bpm_array(x, 125, 10, filter=False)
                clean.assert_not_called()

    def test_serial_parallel_hr_equal(self):
        from evaluation.run_eval import compute_heart_rates
        from metrics import ecg_bpm_array
        length = 1250
        t = np.arange(length)
        x = sum(np.exp(-0.5 * ((t - peak) / 2.) ** 2) for peak in range(90, length - 50, 125))
        gt = np.stack([x, x, x, x])[..., None].astype(np.float64)
        fake = np.roll(gt, 3, axis=1)
        expected = (ecg_bpm_array(gt, 125, 10), ecg_bpm_array(fake, 125, 10, filter=True))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            np.save(Path(directory) / 'overall_gt_data.npy', gt)
            np.save(Path(directory) / 'overall_fake_data.npy', fake)
            for workers in (1, 2):
                actual = compute_heart_rates(Path(directory), 125, workers, chunk_size=2)
                for left, right in zip(expected, actual):
                    np.testing.assert_array_equal(left, right)


if __name__ == '__main__':
    unittest.main()
