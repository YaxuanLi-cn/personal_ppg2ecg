# UniCardio as a PPG→ECG baseline on our MIMIC-IV data

Upstream: https://github.com/thu-ml/UniCardio — *Versatile cardiovascular signal
generation with a unified diffusion transformer*, Nature Machine Intelligence 8, 6–19 (2026).
Cloned at `../` (commit from `git log`), `base_model/` is upstream code.

Everything here reuses **our** data and **our** evaluation code:

* data: `mimic-iv-aligned-ppg_ecgII-processed-filtered/{train,test}.pt`, 1250 samples @125 Hz,
  same global train-set z-score as `v0/utils/ppgecg_dataset.py`; run `python -B ppg2ecg/check_protocol.py`
  to check exact equality across both splits.
* test order: `test.pt` order, `shuffle=False`, so row *i* here is row *i* of
  `v0/results/rectified_flow_{baseline,personal}/.../samples/*.npy`.
* metrics: `v0/evaluation/run_eval.py` imported unmodified.

## Commands

```bash
cd /root/personal_ppg2ecg/UniCardio
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate /venv/ppg

# train the same scored task as PSPFlow: every update predicts ECG from current PPG
python -B ppg2ecg/train.py --save_dir results/unicardio --max_hours 24 --task_mode translation

# recommended: automatically resume and finish exactly one train.pt epoch
bash ppg2ecg/run_one_epoch.sh

# generate on the fixed 20k test subset (≈17 min), or drop --indices for all 178,830 (≈2.5 h)
python -B ppg2ecg/sample.py --ckpt results/unicardio/checkpoints/final.pt \
    --out_dir results/unicardio/samples_20k --indices ppg2ecg/eval_subset_20k.npy

# evaluate; use the same --indices for the PPGFlowECG dirs so all methods see identical rows
python -B ppg2ecg/eval_subset.py --samples_dir results/unicardio/samples_20k
python -B ppg2ecg/eval_subset.py --indices ppg2ecg/eval_subset_20k.npy \
    --samples_dir /root/personal_ppg2ecg/v0/results/rectified_flow_baseline/mimic-iv-waveform/samples
```

The currently available 20k-step checkpoint has been generated and evaluated on the fixed
20,000-row subset. Its metrics and the v4 result on those exact same rows are recorded in
`results/unicardio/eval_20k/comparison.json`; this is a diagnostic comparison, not a final
baseline result, because the training checkpoint is incomplete.

## Deviations from upstream (all forced, all listed)

1. **Trained from scratch, pretrained weights unusable.** `diff_CSDI` has
   `LayerNorm([channels, length/4])` (`diffusion_model_no_compress_final.py:141-144`), so the
   released `no_compress799.pth` is locked to a 500-sample unit length. Our windows are 1250,
   so the checkpoint cannot be loaded without resampling our data — which the comparison forbids.
2. **No BP modality.** Upstream slots are `[PPG | BP | ECG | placeholder]`. Slot B is zero-filled
   and never used as a condition or target; `model_ppg2ecg.py` removes the BP branches from the
   task sampler and keeps the architecture, attention masks, diffusion schedule and loss
   expression untouched. This also means the stage-2/3 curriculum (two/three conditions) is
   unreachable, so training stays in upstream stage 1.
3. **Task mode.** The comparison uses `--task_mode translation`: every optimizer update trains
   the scored conditional task PPG→ECG. This matches PSPFlow's target task; PSPFlow's additional
   representation and patient-context losses are method-specific components, not other prediction
   directions. `--task_mode unified` instead reproduces upstream stage 1 with BP branches removed
   (PPG↔ECG translation plus self-denoising/imputation); only 25% of its updates are PPG→ECG.
   That mode is a separate multi-task ablation and must not be mixed with the primary baseline.
4. **Compute budget.** Upstream trains 800 epochs. At 1250-sample units the sequence is 5000
   tokens and one step at batch 16 costs 0.757 s on the RTX 5090, so PPGFlowECG's equivalent
   budget (100k steps × bs 256 = 25.6 M sample presentations) would take ≈13 days. Training is
   wall-clock capped at 24 h (≈114k steps × bs 16 ≈ 1.8 M presentations, ≈1.1 epochs).
   **Report this budget in the paper; UniCardio is under-trained relative to upstream.**
   Batch 16 is upstream's *per-GPU* batch, which keeps their `sum()/L` loss scale and `lr=1e-3`
   exactly valid; the LR milestones stay at 18 % / 75 % of `--total_steps`.
5. **Sampling.** DDIM with 6 steps (5 NFE), `n_samples=1`, matching `sample_steps=6` in upstream
   `test_final.py` and comparable to our RF's 10 Euler steps. Upstream additionally averages 50
   independent draws; we do not, because our evaluation protocol scores a single draw and
   averaging would inflate MAE/RMSE while distorting FD/FID.
6. **One upstream edit**: removed the unused `import mat73` from
   `base_model/diffusion_model_no_compress_final.py` (the package is not installed and the module
   is never used). No functional change.

## Existing run status

The checked-in local artifacts are an incomplete run: `last.pt` is a translation checkpoint at
20,000 steps, while the CSV log continues to 24,200 steps; later steps were not checkpointed.
The 20k test-subset samples and diagnostic metrics are saved under `results/unicardio/`. Resume
from the matching translation checkpoint with `--resume results/unicardio/checkpoints/last.pt`
to continue; resume now rejects a checkpoint whose task mode or window length differs from the
requested run. Sampling requires a fresh output directory so a previous subset-index file cannot
silently mislabel a later full-test generation.
