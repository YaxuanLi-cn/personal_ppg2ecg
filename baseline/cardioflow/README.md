# CardioFlow baseline (Nambu et al., ICASSP 2025, T=10)

This is a self-contained implementation of the CardioFlow PPG-to-ECG baseline
from *CardioFlow: Learning to Generate ECG from PPG with Rectified Flow*
([Nambu, Kohjima and Yamamoto, ICASSP 2025](https://doi.org/10.1109/ICASSP49660.2025.10888856)).
It uses the repository's processed MIMIC-IV waveform windows: normalized PPG and
ECG tensors with shape `[N,1250,1]` at 125 Hz.

CardioFlow retains the straight noise-to-ECG flow-matching objective and ten
uniform Euler steps. Its peak-aware condition contains PPG and its first
difference plus a local PPG peak mask. The regression loss gives additional
weight to ECG peak regions and the corresponding PPG peak regions, preserving
QRS morphology during fast sampling. No v0--v4 or latent-model code is
imported.

Run from the repository root with `/venv/ppg/bin/python`:

```bash
python -B baseline/cardioflow/train.py \
  --output baseline/cardioflow/results/cardioflow_t10 \
  --data-dir mimic-iv-aligned-ppg_ecgII-processed-filtered \
  --steps 100000 --batch-size 64 --workers 4

python -B baseline/cardioflow/sample.py \
  --checkpoint baseline/cardioflow/results/cardioflow_t10/last.pt \
  --output baseline/cardioflow/results/cardioflow_t10/test_samples \
  --data-dir mimic-iv-aligned-ppg_ecgII-processed-filtered \
  --batch-size 64 --workers 4

python -u v4/evaluation/run_eval.py \
  --samples_dir baseline/cardioflow/results/cardioflow_t10/test_samples \
  --sampling_rate 125
```

For a short smoke run, pass `--max-samples 4 --steps 2 --workers 0 --width 16
--time-dim 32`. The output arrays and metadata have the same names and shapes
as the existing Rectified Flow baseline. Peak weighting can be reproduced or
ablated with `--peak-weight` and `--ppg-peak-weight`.
