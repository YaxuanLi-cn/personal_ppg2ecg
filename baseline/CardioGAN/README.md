# CardioGAN baseline

This is a PyTorch adaptation of CardioGAN (Sarkar & Etemad, AAAI 2021) for
the repository's MIMIC-IV PPG-to-ECG task. The public implementation is
TensorFlow-only and assumes 4-second, 128 Hz (512-point) windows. This adapter
supports the existing globally normalized 125 Hz, 1250-point tensors without
resampling the evaluation data.

It retains the paper's attention U-Net generators, time and spectrogram
discriminators, adversarial losses, and cycle L1 loss. Training uses
independent PPG and ECG batches by default, as in the original unpaired
CardioGAN. `--paired` is reserved for a paired ablation in the command-line
interface and is recorded in the checkpoint metadata.

```bash
cd /root/personal_ppg2ecg/baseline/CardioGAN
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate /venv/ppg
python -B train.py --steps 10 --batch-size 4 --num-workers 0 --out-dir results/smoke
python -B train.py --steps 100000 --out-dir results/cardiogan
python -B sample.py --ckpt results/cardiogan/final.pt \
  --out-dir results/cardiogan/samples_20k \
  --indices ../../UniCardio/ppg2ecg/eval_subset_20k.npy
python -B evaluate.py --samples-dir results/cardiogan/samples_20k --sampling-rate 125
```

The evaluator is `v4/evaluation/run_eval.py`, so it reports MAE, RMSE, FD,
MAE_hr_paired, and MAE_hr_group with the same protocol as v4. FID is omitted
when the ECGFounder checkpoint is unavailable.
