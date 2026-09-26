# CLEP-GAN baseline

This folder contains a PyTorch adaptation of the paper's CLEP-GAN (Attention
U-Net variant) for the repository's paired MIMIC-IV waveform tensors. It keeps
the paper's two Attention U-Nets, two-level symmetric cross-modal NT-Xent loss,
PPG/ECG self reconstruction, PPG-to-ECG reconstruction, and ECG time/STFT
discriminators. Inference uses only the PPG encoder and ECG decoder.

The implementation is based on the official repository at
<https://github.com/Mathematics-Analytics-Data-Science-Lab/CLEP-GAN>, commit
`9b35d058c40bb7cf2a3007e8180a467ccd46de3e`, and the paper at
<https://doi.org/10.1186/s12859-025-06276-0>. The upstream experiment targets
BIDMC/CapnoBase and 128-sample segments. Here the model is adapted to the
repository's normalized 125 Hz, 1250-sample MIMIC-IV windows. We use a seeded
5% validation holdout from `train.pt` (up to 2,048 windows), select checkpoints
by validation MAE+RMSE, then evaluate once on the full fixed test split.

```bash
cd /root/personal_ppg2ecg
/venv/ppg/bin/python baseline/CLEP-GAN/train.py \
  --data-dir mimic-iv-aligned-ppg_ecgII-processed-filtered \
  --out-dir baseline/CLEP-GAN/results/clepgan --steps 50000 --batch-size 16
/venv/ppg/bin/python baseline/CLEP-GAN/sample.py \
  --ckpt baseline/CLEP-GAN/results/clepgan/best.pt \
  --data-dir mimic-iv-aligned-ppg_ecgII-processed-filtered \
  --out-dir baseline/CLEP-GAN/results/clepgan/test
/venv/ppg/bin/python baseline/CLEP-GAN/evaluate.py \
  --samples-dir baseline/CLEP-GAN/results/clepgan/test
```

`comparison_with_v4.json` records the shared five metrics on identical test
targets and ground truth. The result is a within-subject MIMIC comparison; it
does not reproduce CLEP-GAN's subject-independent BIDMC/CapnoBase experiment.
The common evaluator's legacy heart-rate caveat is recorded in that report.
