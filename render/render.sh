#!/bin/bash
# Render SMPLX body motion from pkl files.
#
# Prerequisites:
#   export MUJOCO_GL='osmesa'
#   export PYOPENGL_PLATFORM='osmesa'
#
# Usage:
#   # Render ground truth motion
#   bash render.sh --pkl_file path/to/data.pkl --save_path output.mp4
#
#   # Render with audio overlay
#   bash render.sh --pkl_file path/to/data.pkl --audio path/to/audio.wav --save_path output.mp4
#
#   # Render model predictions (ours)
#   bash render.sh --pkl_file path/to/pred.pkl --gt_file path/to/gt.pkl --mode ours --save_path output.mp4

export MUJOCO_GL='osmesa'
export PYOPENGL_PLATFORM='osmesa'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python "${SCRIPT_DIR}/render_streamer.py" "$@"
