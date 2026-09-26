"""UniCardio restricted to the two modalities our dataset actually has.

The upstream ``CSDI_base.one_condition`` samples a conditioning modality out of
{A=PPG, B=BP, C=ECG}.  Our data has no BP, so slot B is a zero vector and any
task that conditions on it or predicts it is pure noise.  This subclass keeps
the architecture, the diffusion schedule, the attention masks and the loss
expression untouched and only removes the BP branches from the task sampler.

Two task modes:
  ``unified``     -- upstream stage-1 curriculum minus BP: cross-modality
                     (PPG->ECG and ECG->PPG) plus the self-conditioning
                     denoise/impute tasks on PPG and ECG.
  ``translation`` -- only the PPG->ECG cross-modality objective.  Strictly
                     stronger for the metric we report, given a fixed budget.
"""
import random

import torch

from diffusion_model_no_compress_final import CSDI_base


class UniCardioPPG2ECG(CSDI_base):
    def __init__(self, config, device, L, task_mode="translation"):
        super().__init__(config, device, L=L)
        assert task_mode in ("unified", "translation")
        self.task_mode = task_mode

    def _sample_t_alpha(self, B, device, is_train, set_t):
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(device)
        return t, self.alpha_torch[t]

    def translation_loss(self, observed_data, is_train=1, set_t=-1):
        """PPG (slot A) conditions, predict the noise added to ECG (slot C)."""
        B, K, L = observed_data.shape
        device = observed_data.device
        t, current_alpha = self._sample_t_alpha(B, device, is_train, set_t)
        q = int(L / 4)
        noise = torch.randn(B, K, q).to(device)

        noisy_C = (current_alpha ** 0.5) * observed_data[:, :, 2 * q:3 * q] \
            + (1.0 - current_alpha) ** 0.5 * noise
        inp = torch.concat([observed_data[:, :, 0:q], noise, noisy_C, noise], dim=-1)
        predicted = self.diffmodel(inp, t, self.mask1.to(self.device), mode=2, borrow_mode=2)
        residual = noise - predicted
        return (residual ** 2).sum() / L

    def one_condition(self, observed_data, sig_impute, sig_denoise, mask,
                      task_dice, dirty_dice, is_train=1, set_t=-1):
        """Upstream one-condition stage with the BP (slot B) branches removed."""
        B, K, L = observed_data.shape
        device = observed_data.device
        t, current_alpha = self._sample_t_alpha(B, device, is_train, set_t)
        q = int(L / 4)
        noise = torch.randn(B, K, q).to(device)

        if task_dice > 0.5:  # cross-modality
            if random.randint(1, 2) == 1:
                # A (PPG) conditions -> predict C (ECG)
                noisy_C = (current_alpha ** 0.5) * observed_data[:, :, 2 * q:3 * q] \
                    + (1.0 - current_alpha) ** 0.5 * noise
                inp = torch.concat([observed_data[:, :, 0:q], noise, noisy_C, noise], dim=-1)
                predicted = self.diffmodel(inp, t, self.mask1.to(self.device), mode=2, borrow_mode=2)
            else:
                # C (ECG) conditions -> predict A (PPG)
                noisy_A = (current_alpha ** 0.5) * observed_data[:, :, 0:q] \
                    + (1.0 - current_alpha) ** 0.5 * noise
                inp = torch.concat([noisy_A, noise, observed_data[:, :, 2 * q:3 * q], noise], dim=-1)
                predicted = self.diffmodel(inp, t, self.mask3.to(self.device), mode=0, borrow_mode=0)
            return ((noise - predicted) ** 2).sum() / L

        # self-conditioning (denoising / imputation) on A or C
        if dirty_dice > 0.5:
            dirty_sig = sig_impute
        else:
            dirty_sig = sig_denoise
            mask = torch.ones_like(mask)

        if random.randint(1, 2) == 1:
            noisy_D = (current_alpha ** 0.5) * observed_data[:, :, 0:q] \
                + (1.0 - current_alpha) ** 0.5 * noise
            inp = torch.concat([dirty_sig[:, :, 0:q], noise, noise, noisy_D], dim=-1)
            predicted = self.diffmodel(inp, t, self.mask1.to(self.device), mode=3, borrow_mode=0)
        else:
            noisy_D = (current_alpha ** 0.5) * observed_data[:, :, 2 * q:3 * q] \
                + (1.0 - current_alpha) ** 0.5 * noise
            inp = torch.concat([noise, noise, dirty_sig[:, :, 2 * q:3 * q], noisy_D], dim=-1)
            predicted = self.diffmodel(inp, t, self.mask3.to(self.device), mode=3, borrow_mode=2)
        return (((noise - predicted) * mask) ** 2).sum() / L

    def trainning(self, observed_data, sig_impute, sig_denoise, mask, task_dice,
                  dirty_dice, condition_dice, train_threshold, stage,
                  is_train=1, set_t=-1):
        if self.task_mode == "translation":
            return self.translation_loss(observed_data, is_train=is_train, set_t=set_t)
        # ``unified``: stage 1 of the upstream curriculum (one condition).
        # Stages 2/3 need a third modality, which our data does not have.
        return self.one_condition(observed_data, sig_impute, sig_denoise, mask,
                                  task_dice, dirty_dice, is_train=is_train, set_t=set_t)
