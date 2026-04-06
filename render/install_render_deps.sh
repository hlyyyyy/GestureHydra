#!/bin/bash
set -euo pipefail

SKIP_SYSTEM=0

for arg in "$@"; do
  case "$arg" in
    --skip-system)
      SKIP_SYSTEM=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: bash render/install_render_deps.sh [--skip-system]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please activate the target conda environment first, e.g. conda activate gesturehydra" >&2
  exit 1
fi

PIP_INDEX_ARGS=(-i https://pypi.org/simple)

python -m pip install "${PIP_INDEX_ARGS[@]}" --upgrade pip
python -m pip install "${PIP_INDEX_ARGS[@]}"   'setuptools==80.9.0' 'wheel==0.45.1'
python -m pip install "${PIP_INDEX_ARGS[@]}"   'numpy<2' 'opencv-python<4.10'   'transformers==4.30.2' librosa scipy smplx easydict webdataset tqdm   pydub praat-parselmouth packaging 'PyYAML>=6.0' tensorboard matplotlib   pyrender trimesh 'pyglet<2'   'requests==2.32.3' 'charset_normalizer==3.3.2' 'chardet==5.2.0'
python -m pip uninstall -y PyOpenGL || true
python -m pip install "${PIP_INDEX_ARGS[@]}" --no-build-isolation   git+https://github.com/mmatl/pyopengl.git

if [[ "$SKIP_SYSTEM" == "0" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y       ffmpeg libglu1-mesa
  else
    echo "apt-get not found; please install ffmpeg and libglu1-mesa manually." >&2
  fi
fi

echo "Render dependencies are installed in: ${CONDA_PREFIX}"
echo "Pinned packaging tools: setuptools==80.9.0 wheel==0.45.1"
