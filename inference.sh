#!/bin/bash
# Inference script for GestureHydra.
#
# Usage:
#   bash inference.sh [CONFIG] [CHECKPOINT] [OUT_DIR]
#
# Example:
#   bash inference.sh configs/gesturehydra/infer.py work_dirs/stage2.pth work_dirs/style_infer_results/
#
# All variables can be overridden via environment, e.g.:
#   DEVICE=cpu INPUT_CSV=data/datasets/streamer/smplx_wav_va_test_unseen.csv bash inference.sh
#
# Environment:
#   conda activate gesturehydra

set -euo pipefail

CONFIG=${1:-configs/gesturehydra/infer.py}
CHECKPOINT=${2:-work_dirs/model.pth}
OUT_DIR=${3:-work_dirs/style_infer_results_anon/}
INPUT_CSV=${INPUT_CSV:-data/datasets/streamer/smplx_wav_va_test_seen.csv}
STYLE_CSV=${STYLE_CSV:-data/datasets/streamer/smplx_wav_va_train.csv}
SEED_LEN=${SEED_LEN:-5}
DEVICE=${DEVICE:-cuda}

PYTHONPATH=".:${PYTHONPATH:-}" python tools/inference.py \
  "$CONFIG" \
  "$CHECKPOINT" \
  --out "$OUT_DIR" \
  --input-csv "$INPUT_CSV" \
  --style-csv "$STYLE_CSV" \
  --seed-len "$SEED_LEN" \
  --device "$DEVICE"
