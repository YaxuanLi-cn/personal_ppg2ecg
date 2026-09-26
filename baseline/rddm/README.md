# RDDM (T=10)

This implementation follows Shome, Sarkar and Etemad, *Region-Disentangled
Diffusion Model for High-Fidelity PPG-to-ECG Translation* (AAAI 2024). It uses
the paper's ROI-guided forward process, two denoisers (`epsilon` for ROI noise
and `rho` for the ROI-guided signal), linear DDPM variance schedule, and ten
reverse diffusion steps. ECG local peak masks are used only during training;
sampling is conditioned on PPG alone. The paper's schedule is β∈[0.0001,0.2],
ROI width Γ=32, and loss coefficients λ1=100, λ2=1.

```bash
/venv/ppg/bin/python -B baseline/rddm/train.py --output baseline/rddm/results/rddm_t10 --steps 100000 --batch-size 64 --workers 4
/venv/ppg/bin/python -B baseline/rddm/sample.py --checkpoint baseline/rddm/results/rddm_t10/last.pt --output baseline/rddm/results/rddm_t10/test_samples
/venv/ppg/bin/python -B v4/evaluation/run_eval.py --samples_dir baseline/rddm/results/rddm_t10/test_samples --sampling_rate 125
```
