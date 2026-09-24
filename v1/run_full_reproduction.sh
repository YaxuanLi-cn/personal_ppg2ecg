#!/bin/bash
# Full reproduction orchestrator for personal_ppg2ecg.
# Runs all four training stages sequentially (single GPU):
#   1. CardioAlign VAE        (40k iters)
#   2. IBExtractor            (40k iters)
#   3. Rectified Flow baseline (pf,  100k steps)
#   4. Rectified Flow personal (ppf, 100k steps)
# Data preprocessing (Arrow -> filtered npz -> train.pt/test.pt) is assumed done.
set -e

PY=${PY:-/venv/ppg/bin/python}
cd "$(dirname "$0")"
LOG=${LOG:-"$PWD/logs"}
STATUS=$LOG/STATUS.txt
mkdir -p "$LOG"

stamp() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

stamp "=== Reproduction run started ==="

stamp "STAGE 1/4: CardioAlign VAE (40k)"
$PY -u model/cardioalign_encoder/train.py \
    --config config/cardioalign_encoder.yaml \
    --save_dir results/cardioalign_encoder > "$LOG/train_vae.log" 2>&1
stamp "STAGE 1/4 DONE: VAE"

stamp "STAGE 2/4: IBExtractor (40k)"
$PY -u model/Individual_base_extractor/train.py \
    --config config/ib_extractor.yaml \
    --save_dir results/IBExtractor > "$LOG/train_ibe.log" 2>&1
stamp "STAGE 2/4 DONE: IBExtractor"

stamp "STAGE 3/4: Rectified Flow baseline (pf, 100k)"
$PY -u main.py --train \
    --config_file config/latent_rectified_flow.yaml \
    --model_type pf --name latent_rectified_flow --output baseline > "$LOG/train_rf_baseline.log" 2>&1
stamp "STAGE 3/4 DONE: RF baseline"

stamp "STAGE 4/4: Rectified Flow personal (ppf, 100k)"
$PY -u main.py --train \
    --config_file config/personal_latent_rectified_flow.yaml \
    --model_type ppf --name personal_latent_rectified_flow --output personal > "$LOG/train_rf_personal.log" 2>&1
stamp "STAGE 4/4 DONE: RF personal"

stamp "=== ALL TRAINING COMPLETE ==="
