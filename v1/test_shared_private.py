import copy
import gc
import importlib.util
import logging
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from engine.solver import Trainer
from model.cardioalign_encoder.cardioalign_model import (
    ECGDecoder, ECGPrivateEncoder, VAE_Decoder, VAE_Encoder,
    loss_function, shared_private_decorrelation,
)
from model.cardioalign_encoder.train import _infonce_loss, _kl_gaussians, train_loop
from model.Individual_base_extractor.ib_extractor import IBExtractor
from model.latent_rectified_flow.rectified_flow import PersonalRectifiedFlow, RectifiedFlow
from utils.io_utils import instantiate_from_config, load_yaml_config, merge_opts_to_config


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIGURATIONS = (
    ('baseline', 'cardioalign_encoder.yaml', 'personal_latent_rectified_flow.yaml'),
    ('full', 'cardioalign_shared_private.yaml', 'personal_private_rectified_flow.yaml'),
    ('no_decor', 'cardioalign_shared_private_no_decor.yaml', 'personal_private_rectified_flow_no_decor.yaml'),
)


class SharedPrivateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True

    @classmethod
    def tearDownClass(cls):
        torch.backends.cudnn.deterministic = cls.original_deterministic

    def setUp(self):
        torch.manual_seed(123)

    def tearDown(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_private_shapes_and_decoder_gradients(self):
        shared_encoder = VAE_Encoder().to(DEVICE).eval()
        ecg = torch.randn(2, 1250, 1, device=DEVICE)
        with torch.no_grad():
            shared = shared_encoder(ecg)[0]
        for dim in (1, 3, 8):
            with self.subTest(private_dim=dim):
                private_encoder = ECGPrivateEncoder(private_dim=dim).to(DEVICE)
                private = private_encoder(ecg)
                self.assertEqual(private.shape, (2, dim, shared.shape[-1]))
                self.assertEqual(private_encoder(ecg[:, :1000]).shape, (2, dim, 40))
                decoder = ECGDecoder(private_dim=dim).to(DEVICE)
                s = shared.detach().requires_grad_()
                output = decoder(s, private)
                self.assertEqual(output.shape, ecg.shape)
                loss = (output - ecg).square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(output).all())
                for grad in (s.grad, private_encoder.blocks[-1].weight.grad, decoder.fusion.weight.grad):
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)
                with self.assertRaisesRegex(ValueError, 'matching batch and temporal'):
                    decoder(s, private[..., :-1])
                print('shape check:', 'ECG', tuple(ecg.shape), 's_ecg', tuple(s.shape),
                      'r_ecg', tuple(private.shape), 'decoder input', (2, 4 + dim, 50),
                      'ECG reconstruction', tuple(output.shape))
                del private_encoder, private, decoder, s, output, loss

    def test_alignment_has_no_private_gradient(self):
        mu_ppg = torch.randn(4, 4, 50, requires_grad=True)
        mu_ecg = torch.randn(4, 4, 50, requires_grad=True)
        log_ppg = torch.zeros_like(mu_ppg, requires_grad=True)
        log_ecg = torch.zeros_like(mu_ecg, requires_grad=True)
        private = torch.randn(4, 7, 50, requires_grad=True)
        alignment = (mu_ppg - mu_ecg).square().mean() + .5 * (
            _kl_gaussians(mu_ppg, log_ppg, mu_ecg, log_ecg)
            + _kl_gaussians(mu_ecg, log_ecg, mu_ppg, log_ppg)
        ) + _infonce_loss(mu_ppg, mu_ecg)
        self.assertIsNone(torch.autograd.grad(alignment, private, allow_unused=True)[0])

    def test_decorrelation_formula_and_edge_cases(self):
        for batch in (1, 6):
            s = torch.randn(batch, 4, 9, requires_grad=True)
            r = torch.randn(batch, 7, 9, requires_grad=True)
            pooled_s, pooled_r = s.mean(-1), r.mean(-1)
            norm_s = torch.nn.functional.normalize(pooled_s - pooled_s.mean(0), dim=-1)
            norm_r = torch.nn.functional.normalize(pooled_r - pooled_r.mean(0), dim=-1)
            expected = ((norm_s.T @ norm_r / batch) ** 2).sum()
            actual = shared_private_decorrelation(s, r)
            torch.testing.assert_close(actual, expected)
            actual.backward()
            self.assertTrue(torch.isfinite(s.grad).all() and torch.isfinite(r.grad).all())
            if batch > 1:
                self.assertGreater(s.grad.abs().sum().item(), 0)
                self.assertGreater(r.grad.abs().sum().item(), 0)
        s = torch.ones(4, 4, 9, requires_grad=True)
        r = torch.ones(4, 7, 9, requires_grad=True)
        self.assertEqual(shared_private_decorrelation(s, r).item(), 0.0)

    def test_private_flow_equation_and_conditions(self):
        flow = PersonalRectifiedFlow(
            feature_size=4, temporal_size=50, d_model=256, n_layer_enc=1,
            train_num_points=1, use_shared_private=True, private_dim=7,
        ).to(DEVICE)
        target = torch.randn(2, 7, 50, device=DEVICE)
        shared = torch.randn(2, 4, 50, device=DEVICE)
        patient = torch.randn(2, 256, device=DEVICE)
        noise = torch.randn_like(target)
        time = torch.tensor([0.25, 0.75], device=DEVICE)
        xt = (1 - time[:, None, None]) * noise + time[:, None, None] * target
        velocity = target - noise
        with patch('torch.rand', return_value=time), patch('torch.randn_like', return_value=noise):
            with patch.object(flow, 'output', return_value=velocity) as output:
                loss = flow(target, target=target, cond=shared, personal_cond=patient)
                torch.testing.assert_close(output.call_args.args[0], xt)
                self.assertEqual(loss.item(), 0.0)
        predicted = flow.output(xt, time, cond=shared, personal_cond=patient)
        self.assertEqual(predicted.shape, target.shape)
        predicted.square().mean().backward()
        self.assertGreater(flow.model.personal_emb.weight.grad.abs().sum().item(), 0)
        self.assertGreater(flow.model.condition_emb.sequential[1].weight.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, 'both shared PPG'):
            flow.output(xt, time, cond=shared)
        with self.assertRaisesRegex(ValueError, 'PersonalRectifiedFlow'):
            RectifiedFlow(feature_size=4, temporal_size=50, d_model=256, use_shared_private=True)
        print('flow shape check:', 's_ppg', tuple(shared.shape), 'e_patient', tuple(patient.shape),
              'flow x_t', tuple(xt.shape), 'condition tokens', (2, 50, 256),
              'velocity', tuple(predicted.shape))

    def test_boolean_config_overrides(self):
        config = {'model': {'use_shared_private': True}, 'train': {'use_decorrelation': True}}
        merge_opts_to_config(config, ['model.use_shared_private', 'false', 'train.use_decorrelation', 'false'])
        self.assertIs(config['model']['use_shared_private'], False)
        self.assertIs(config['train']['use_decorrelation'], False)
        with self.assertRaises(ValueError):
            merge_opts_to_config(config, ['model.use_shared_private', 'invalid'])

    def test_baseline_transformer_checkpoint_compatibility(self):
        original = ROOT.parent / 'v0/model/latent_rectified_flow/transformer.py'
        if not original.exists():
            self.skipTest('v0 reference tree not available')
        spec = importlib.util.spec_from_file_location('v0_transformer_reference', original)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        old = module.PersonalTransformer(n_embd=256, n_layer_enc=1, max_len=50).eval()
        new = PersonalRectifiedFlow(feature_size=4, temporal_size=50, d_model=256, n_layer_enc=1).model.eval()
        new.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        x, shared = torch.randn(2, 4, 50), torch.randn(2, 4, 50)
        patient, t = torch.randn(2, 256), torch.rand(2)
        with torch.no_grad():
            torch.testing.assert_close(old(x, t, cond=shared, personal_cond=patient),
                                       new(x, t, cond=shared, personal_cond=patient), rtol=0, atol=0)

    def test_preserved_modules_are_byte_identical(self):
        original = ROOT.parent / 'v0'
        if not original.exists():
            self.skipTest('v0 reference tree not available')
        paths = [
            'model/Individual_base_extractor/ib_extractor.py',
            'model/Individual_base_extractor/train.py', 'config/ib_extractor.yaml',
            'utils/ppgecg_dataset.py', 'utils/ppgecg_dataset_.py', 'utils/saved_dataset_.py',
        ]
        paths += [str(p.relative_to(ROOT)) for folder in ('evaluation', 'data_process_to_npz')
                  for p in (ROOT / folder).glob('*.py')]
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual((ROOT / path).read_bytes(), (original / path).read_bytes())

    @unittest.skipUnless(torch.cuda.is_available(), 'Existing main.py requires CUDA')
    def test_cli_round_trip_with_unequal_latent_dimensions(self):
        with tempfile.TemporaryDirectory(prefix='cli-verification-', dir=ROOT) as temp:
            temp = Path(temp)
            dataset = {'PPG': torch.randn(4, 1250), 'ECG': torch.randn(4, 1250),
                       'file_name': torch.tensor([1, 1, 2, 2])}
            for split in ('train', 'test'):
                torch.save(dataset, temp / f'{split}.pt')
            (temp / 'z_score_mean_std.json').write_text(json.dumps({
                'train': {'PPG': {'mean': 0., 'std': 1.}, 'ECG': {'mean': 0., 'std': 1.}},
            }))
            config1 = load_yaml_config(str(ROOT / 'config/cardioalign_shared_private.yaml'))
            config1['train'].update(total_iterations=1, save_interval=1, log_interval=1,
                                    batch_size=4, num_workers=0, pin_memory=False,
                                    save_dir=str(temp / 'stage1'))
            config1['model']['private_dim'] = 3
            config1['data']['train_dir'] = str(temp)
            stage1_config = temp / 'stage1.yaml'
            stage1_config.write_text(yaml.safe_dump(config1))
            result = subprocess.run(
                [sys.executable, '-B', 'model/cardioalign_encoder/train.py', '--config', str(stage1_config)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stdout[-6000:])
            patient_path = temp / 'patient.pth'
            torch.save({'model': IBExtractor().state_dict()}, patient_path)
            config2 = load_yaml_config(str(ROOT / 'config/personal_private_rectified_flow.yaml'))
            config2['model']['params']['private_dim'] = 3
            config2['solver']['vae']['checkpoint'] = str(temp / 'stage1/mimic-iv-waveform/checkpoints/VAE-iter-1.pth')
            config2['solver']['ibe']['checkpoint'] = str(patient_path)
            config2['solver'].update(max_steps=1, save_cycle=1, results_folder=str(temp / 'flow'))
            config2['data']['train'].update(dir=str(temp), batch_size=4, num_workers=0, pin_memory=False)
            config2['data']['test'].update(saved_dir=str(temp), batch_size=4, num_workers=0, pin_memory=False)
            stage2_config = temp / 'stage2.yaml'
            stage2_config.write_text(yaml.safe_dump(config2))
            base_command = [sys.executable, '-B', 'main.py', '--model_type', 'ppf', '--config_file', str(stage2_config)]
            for options in (['--train'], ['--milestone', '1', 'sample.sampling_steps', '5', 'sample.num_samples', '2']):
                result = subprocess.run(base_command + options, cwd=ROOT, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout[-6000:])
            output_dir = temp / 'flow/samples'
            samples = np.load(output_dir / 'overall_fake_samples.npy')
            self.assertEqual(samples.shape, (4, 2, 1250, 1))
            self.assertTrue(np.isfinite(samples).all())
            np.testing.assert_array_equal(np.load(output_dir / 'overall_fake_data.npy'), samples[:, 0])
            print('CLI Stage 1 -> Stage 2 -> generation passed with shared_dim=4, private_dim=3:', samples.shape)

    def test_three_configurations_end_to_end(self):
        for name, stage1_file, stage2_file in CONFIGURATIONS:
            with self.subTest(configuration=name):
                self._run_configuration(name, stage1_file, stage2_file)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _run_configuration(self, name, stage1_file, stage2_file):
        config1 = load_yaml_config(str(ROOT / 'config' / stage1_file))
        config2 = load_yaml_config(str(ROOT / 'config' / stage2_file))
        model_cfg, train_cfg = config1['model'], config1['train']
        use_private = model_cfg['use_shared_private']
        self.assertEqual(use_private, config2['model']['params']['use_shared_private'])
        with tempfile.TemporaryDirectory(prefix='verification-', dir=ROOT) as temp:
            shared_encoder = VAE_Encoder().to(DEVICE)
            private_encoder = ECGPrivateEncoder(
                private_dim=model_cfg['private_dim'], hidden_dim=model_cfg['private_hidden_dim'],
            ).to(DEVICE) if use_private else None
            ecg_decoder = (ECGDecoder(private_dim=model_cfg['private_dim'], **model_cfg['decoder'])
                           if use_private else VAE_Decoder(**model_cfg['decoder'])).to(DEVICE)
            ppg_decoder = VAE_Decoder(**model_cfg['decoder']).to(DEVICE)
            modules = [shared_encoder, ecg_decoder, ppg_decoder]
            if private_encoder is not None:
                modules.append(private_encoder)
            optimizer = torch.optim.SGD([p for m in modules for p in m.parameters()], lr=1e-5)
            ppg, ecg, ppg_ref, ecg_ref = (torch.randn(4, 1250) for _ in range(4))
            loader = DataLoader(TensorDataset(ppg, ecg), batch_size=4)
            grad_norms = []
            handles = []
            for module in (shared_encoder.blocks[-1], ecg_decoder.residual_decoder.blocks[-1],
                           ppg_decoder.residual_decoder.blocks[-1]):
                handles.append(module.weight.register_hook(lambda grad: grad_norms.append(float(grad.norm()))))
            if use_private:
                handles.append(private_encoder.blocks[-1].weight.register_hook(
                    lambda grad: grad_norms.append(float(grad.norm()))))
            with patch('model.cardioalign_encoder.train.shared_private_decorrelation',
                       wraps=shared_private_decorrelation) as decor:
                with self.assertLogs('stage1-test', level='INFO') as logs:
                    train_loop(
                        loader, shared_encoder, None, ecg_decoder, ppg_decoder,
                        loss_function, optimizer, None, DEVICE,
                        private_encoder=private_encoder, decoder_config=model_cfg['decoder'],
                        kld_weight=train_cfg['kld_weight'], lambda_align=train_cfg['lambda_align'],
                        lambda_cross=train_cfg['lambda_cross'], lambda_infonce=train_cfg['lambda_infonce'],
                        lambda_decor=train_cfg['lambda_decor'], use_decorrelation=train_cfg['use_decorrelation'],
                        use_fft_loss=True, save_weights_path=temp, logger=logging.getLogger('stage1-test'),
                        total_iterations=1, save_interval=1, log_interval=1,
                    )
                self.assertEqual(decor.call_count, int(use_private and train_cfg['use_decorrelation']))
            for handle in handles:
                handle.remove()
            self.assertEqual(len(grad_norms), len(modules))
            self.assertTrue(all(np.isfinite(x) and x > 0 for x in grad_norms))
            self.assertIn('weighted_decor:', logs.output[0])
            print(name, logs.output[0])
            stage1_path = str(Path(temp) / 'VAE-iter-1.pth')
            del shared_encoder, ecg_decoder, ppg_decoder, private_encoder, modules, optimizer, module
            gc.collect()
            patient = IBExtractor()
            patient_path = str(Path(temp) / 'patient.pth')
            torch.save({'model': patient.state_dict()}, patient_path)
            del patient
            config2['solver']['vae']['checkpoint'] = stage1_path
            config2['solver']['ibe']['checkpoint'] = patient_path
            config2['solver']['results_folder'] = str(Path(temp) / 'flow')
            config2['solver']['max_steps'] = 1
            config2['solver']['save_cycle'] = 1
            config2['solver']['log_frequency'] = 1
            config2['solver']['ema']['update_interval'] = 1
            config2['solver']['scheduler']['params']['warmup'] = 0
            config2['model']['params']['n_layer_enc'] = 1
            config2['model']['params']['train_num_points'] = 1
            flow = instantiate_from_config(config2['model']).to(DEVICE)
            pair_loader = DataLoader(TensorDataset(ppg, ecg, ppg_ref, ecg_ref), batch_size=4)
            trainer = Trainer(config2, SimpleNamespace(model_type='ppf', name=name), flow, pair_loader, Mock())
            frozen = [trainer.vae_encoder_ecg, trainer.vae_encoder_ppg, trainer.vae_decoder_ecg,
                      trainer.vae_decoder_ppg, trainer.ib_extractor]
            if use_private:
                frozen.append(trainer.private_encoder_ecg)
                self.assertIs(trainer.vae_encoder_ecg, trainer.vae_encoder_ppg)
            for module in frozen:
                self.assertFalse(module.training)
                self.assertTrue(all(not p.requires_grad for p in module.parameters()))
            target_encoder = trainer.private_encoder_ecg if use_private else trainer.vae_encoder_ecg
            targets = []
            hook = target_encoder.register_forward_hook(
                lambda module, inputs, output: targets.append((output if use_private else output[0]).detach().clone()))
            with patch.object(flow, 'forward', wraps=flow.forward) as forward:
                trainer.train()
                args = forward.call_args.kwargs
                torch.testing.assert_close(args['target'], targets[0])
                self.assertEqual(args['personal_cond'].shape, (4, 256))
                self.assertEqual(args['cond'].shape, (4, 4, 50))
            hook.remove()
            self.assertEqual(trainer.step, 1)
            for module in frozen:
                self.assertTrue(all(p.grad is None for p in module.parameters()))
            trainer.load(1)
            if use_private:
                mismatch = copy.deepcopy(trainer.representation)
                trainer.representation['private_dim'] += 1
                with self.assertRaisesRegex(ValueError, 'configurations do not match'):
                    trainer.load(1)
                trainer.representation = mismatch
            for steps in (5, 10):
                with patch.object(trainer.ema.ema_model, 'output', wraps=trainer.ema.ema_model.output) as output:
                    with patch.object(trainer.vae_encoder_ppg, 'forward', wraps=trainer.vae_encoder_ppg.forward) as shared_forward:
                        samples = trainer.sample(ppg[:1], ppg_ref=ppg_ref[:1], ecg_ref=ecg_ref[:1],
                                                 sampling_steps=steps, num_samples=2)
                        self.assertEqual(shared_forward.call_count, 1)
                    self.assertEqual(output.call_count, 2 * steps)
                    torch.testing.assert_close(output.call_args_list[0].kwargs['cond'],
                                               output.call_args_list[steps].kwargs['cond'])
                    torch.testing.assert_close(output.call_args_list[0].kwargs['personal_cond'],
                                               output.call_args_list[steps].kwargs['personal_cond'])
                self.assertEqual(samples.shape, (1, 2, 1250, 1))
                self.assertTrue(torch.isfinite(samples).all())
                self.assertFalse(torch.equal(samples[:, 0], samples[:, 1]))
                print(name, 'T', steps, 'generated ECG', tuple(samples.shape))
            with patch.object(trainer.vae_encoder_ppg, 'forward', wraps=trainer.vae_encoder_ppg.forward) as shared_forward:
                torch.manual_seed(11)
                first = trainer.sample(ppg[:1], ecg=torch.full_like(ecg[:1], float('nan')),
                                       ppg_ref=ppg_ref[:1], ecg_ref=ecg_ref[:1], sampling_steps=5)
                torch.manual_seed(11)
                second = trainer.sample(ppg[:1], ppg_ref=ppg_ref[:1], ecg_ref=ecg_ref[:1], sampling_steps=5)
                self.assertEqual(shared_forward.call_count, 2)
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            sample_dir = str(Path(temp) / 'samples')
            with patch.object(trainer.vae_encoder_ppg, 'forward', wraps=trainer.vae_encoder_ppg.forward) as shared_forward:
                samples, real, signals = trainer.sample_shift(pair_loader, shape=[1250, 1], sampling_steps=5,
                                                              num_samples=2, save_dir=sample_dir, subset_save_threshold=2)
                self.assertEqual(shared_forward.call_count, 1)
            self.assertEqual(samples.shape, (4, 2, 1250, 1))
            np.testing.assert_array_equal(np.load(Path(sample_dir) / 'overall_fake_data.npy'), samples[:, 0])
            np.testing.assert_array_equal(np.load(Path(sample_dir) / 'overall_fake_samples.npy'), samples)
            np.testing.assert_array_equal(real, ecg.numpy()[..., None])
            np.testing.assert_array_equal(signals, ppg.numpy()[..., None])
            with self.assertRaisesRegex(ValueError, 'reference PPG and ECG'):
                trainer.sample(ppg)
            with self.assertRaisesRegex(ValueError, 'must be positive'):
                trainer.sample(ppg, ppg_ref=ppg_ref, ecg_ref=ecg_ref, sampling_steps=0)
            del trainer, flow, frozen, target_encoder, targets, args, forward, shared_forward


if __name__ == '__main__':
    unittest.main(verbosity=2)
