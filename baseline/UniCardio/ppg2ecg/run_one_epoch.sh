#!/usr/bin/env bash
set -euo pipefail

# Continue the translation baseline until one complete pass over train.pt.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PPG_PYTHON:-/root/.conda/envs/ppg/bin/python}"
DATA_DIR="${UNICARDIO_DATA_DIR:-/root/personal_ppg2ecg/mimic-iv-aligned-ppg_ecgII-processed-filtered}"
SAVE_DIR="${UNICARDIO_SAVE_DIR:-results/unicardio}"
BATCH_SIZE="${UNICARDIO_BATCH_SIZE:-16}"
MAX_HOURS="${UNICARDIO_MAX_HOURS:-30}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python"
fi

TOTAL_STEPS="$($PYTHON_BIN - "$DATA_DIR" "$BATCH_SIZE" <<'PY'
import sys
import torch

data_dir, batch_size = sys.argv[1], int(sys.argv[2])
data = torch.load(f"{data_dir}/train.pt", map_location="cpu", weights_only=False)
print(int(data["PPG"].shape[0]) // batch_size)
PY
)"

CKPT="$SAVE_DIR/checkpoints/last.pt"
if [[ ! -f "$CKPT" ]]; then
    CKPT="$SAVE_DIR/checkpoints/step-20000.pt"
fi
if [[ ! -f "$CKPT" ]]; then
    echo "No UniCardio checkpoint found under $SAVE_DIR/checkpoints" >&2
    exit 1
fi

CURRENT_STEP="$($PYTHON_BIN - "$CKPT" <<'PY'
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(state.get("step", 0)))
PY
)"

echo "data=$DATA_DIR"
echo "checkpoint=$CKPT (step $CURRENT_STEP)"
echo "target_steps=$TOTAL_STEPS batch_size=$BATCH_SIZE"

if (( CURRENT_STEP >= TOTAL_STEPS )); then
    echo "Training already reached one epoch; nothing to run."
    exit 0
fi

exec "$PYTHON_BIN" -u -B ppg2ecg/train.py \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --resume "$CKPT" \
    --task_mode translation \
    --batch_size "$BATCH_SIZE" \
    --total_steps "$TOTAL_STEPS" \
    --max_hours "$MAX_HOURS" \
    --num_workers "${UNICARDIO_NUM_WORKERS:-8}" \
    --ckpt_every 5000 \
    --log_every 100
