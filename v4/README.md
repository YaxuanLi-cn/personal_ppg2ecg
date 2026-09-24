# v4 — shared/private ECG generation with patient-conditioned private prediction

v4 restores the v1 core mechanism as an isolated implementation:

    current PPG ──► shared encoder ────────────────┐
                                                   ├─► joint decoder ──► ECG
    target ECG ──► shared + private encoder  (train only)
    reference PPG/ECG pair ──► patient encoder ──► private predictor + rectified flow

Only the PPG/ECG *shared* embeddings are aligned. The ECG-private embedding is a
distinct latent predicted from (current PPG shared embedding, patient embedding):
a deterministic conditional mean plus a conditional rectified-flow residual.
The decoder consumes exactly the two embeddings; it never sees target ECG at
inference. Public inference: `ECGPredictor(ppg, ppg_ref, ecg_ref, indices=...)`.

## Layout

- `data.py` — dataset loading, subject-disjoint-aware `split_indices`, deterministic same-subject reference draws.
- `pretrained.py` — frozen v1 `SharedEncoder` (PPG anchor, 4×50) and `IBExtractor` patient encoder (256-d), read-only checkpoints.
- `model.py` — trainable shared refiner (64×250), ECG-private encoder (16×250), joint decoder, private predictor, conditional private flow, losses.
- `train.py` — two stages: `--stage rep` (representation + deterministic private prediction + joint decoding) then `--stage flow` (frozen representation, conditional RF residual). EMA weights, cosine LR, validation-selected `best.pt`/`last.pt`.
- `predict.py` — target-free inference; `mode='sample'` (mean + flow residual, deterministic per-window noise) or `mode='mean'`.
- `evaluate.py` — locked-checkpoint full-test generation + five-metric comparison vs v0/v1.
- `evaluation/run_eval.py` — five legacy metrics (MAE, RMSE, FD, MAE_hr_paired, MAE_hr_group), same protocol as v0–v3.
- `test_v4.py` — `python -B -m unittest -v test_v4`.

## Protocol

- Train split: identical to `v2/results/paired_v2/split.npz` (verified in tests).
- Training/validation references: same-subject windows drawn only from the training-fit pool; validation references never come from validation targets.
- Test references: same-subject windows within the test split, seed 42 — verified identical to the recorded v2/v3 draws.
- Model selection: train.pt validation MAE+RMSE only; test data is never used for training, calibration, or selection (`selection.test_used=False` is enforced).
- Output arrays: `overall_fake_data.npy`, `overall_gt_data.npy`, `overall_gt_ppg_data.npy` `(N,1250,1)`; GT/PPG ordering verified against saved v2 arrays during evaluation.
- Inference noise is deterministic per window index (`fixed_noise`), independent of batching/order.

## Commands

```bash
cd /root/personal_ppg2ecg/v4
/venv/ppg/bin/python -B -m unittest -v test_v4
/venv/ppg/bin/python -B train.py --stage rep  --output results/stage_rep  --steps 20000
/venv/ppg/bin/python -B train.py --stage flow --output results/stage_flow --steps 20000 \
    --init-checkpoint results/stage_rep/best.pt --lr 1e-4
/venv/ppg/bin/python -B evaluate.py --checkpoint results/stage_flow/best.pt --output results/final_test
```

## Known limitations (must be disclosed)

- Train/test subjects overlap (within-subject split); references are random same-subject windows, not causal/ordered history. No unseen-patient or causal claims.
- Frozen v1 encoders previously saw all of train.pt, including the validation holdout — validation is a fine-tuning holdout, not a pristine end-to-end holdout.
- HR metrics keep the legacy caveats (fake-only cleaning at hardcoded 128 Hz, asymmetric masks).
- v0/v1 reference draws were not saved; exact equality of their draws cannot be verified.
