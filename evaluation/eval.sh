#!/bin/bash
# FGD Evaluation Script
# Usage: bash eval.sh <pkl_path> <gt_path> [fgd_model_path]
# Example: bash eval.sh /path/to/predictions /path/to/ground_truth/gestures checkpoints/fgd.pth

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FGD_MODEL=${3:-checkpoints/fgd.pth}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -W ignore fgd.py \
    --pkl_path ${1:?"Please provide prediction pkl path as first argument"} \
    --gt_path ${2:?"Please provide ground truth path as second argument"} \
    --fgd_model "$FGD_MODEL"
