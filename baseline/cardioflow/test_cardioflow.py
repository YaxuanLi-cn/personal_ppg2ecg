import unittest

import torch

from model import CardioFlow, CardioFlowSampler, peak_mask


class CardioFlowSmokeTest(unittest.TestCase):
    def test_peak_mask_and_flow_shapes(self):
        model = CardioFlow(width=16, time_dim=32)
        ppg = torch.randn(2, 1250, 1)
        ecg = torch.randn_like(ppg)
        self.assertEqual(tuple(peak_mask(ppg).shape), tuple(ppg.shape))
        self.assertEqual(tuple(model(ecg, torch.rand(2), ppg).shape), tuple(ppg.shape))
        self.assertTrue(torch.isfinite(model.flow_matching_loss(ppg, ecg)))
        fake = CardioFlowSampler(model)(ppg, torch.randn_like(ppg))
        self.assertEqual(tuple(fake.shape), tuple(ppg.shape))

    def test_sampler_requires_ten_steps(self):
        with self.assertRaises(ValueError):
            CardioFlowSampler(CardioFlow(width=16, time_dim=32), steps=9)


if __name__ == "__main__":
    unittest.main()
