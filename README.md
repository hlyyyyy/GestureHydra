<div align="center">

# GestureHYDRA

### Semantic Co-speech Gesture Synthesis via Hybrid Modality Diffusion Transformer and Cascaded-Synchronized Retrieval-Augmented Generation

[Quanwei Yang<sup>1,*</sup>](#) ·
[Luying Huang<sup>2,*</sup>](#) ·
[Kaisiyuan Wang<sup>2,†</sup>](#) ·
[Jiazhi Guan<sup>2</sup>](#) ·
[Shengyi He<sup>2</sup>](#) ·
[Fengguo Li<sup>2</sup>](#) ·
[Hang Zhou<sup>2</sup>](#) ·
[Lingyun Yu<sup>1</sup>](#) ·
[Yingying Li<sup>2</sup>](#) ·
[Haocheng Feng<sup>2</sup>](#) ·
[Hongtao Xie<sup>1,†</sup>](#)

<sup>1</sup>University of Science and Technology of China
<br>
<sup>2</sup>Baidu Inc.
<br>
<sup>*</sup>Equal contribution.
<sup>†</sup>Corresponding authors.

[![arXiv](https://img.shields.io/badge/arXiv-2507.22731-b31b1b.svg)](https://arxiv.org/abs/2507.22731v2)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://mumuwei.github.io/GestureHYDRA/)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/datasets/mumuwei/Streamer)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg)](https://www.python.org/)

</div>

## Teaser

<div align="center">
  <a href="assets/teaser1.pdf">
    <img src="assets/teaser1.png" alt="GestureHYDRA teaser" width="92%">
  </a>
</div>

GestureHYDRA is a semantic co-speech gesture generation framework that combines a hybrid modality diffusion transformer with a cascaded-synchronized retrieval-augmented generation pipeline, enabling more reliable semantic gesture activation and flexible gesture editing.

## Overview

Human co-speech gestures are not only rhythmic body movements, but also important carriers of explicit semantics. GestureHYDRA focuses on synthesizing gestures with stronger semantic awareness, especially instructional hand gestures such as numbers, directions, greeting, and denial. To support this setting, the project is built around three key components:

- A hybrid modality diffusion transformer for gesture modeling under audio, text, and identity conditions.
- A cascaded-synchronized retrieval-augmented generation pipeline for reliable semantic gesture activation and temporal synchronization.
- The Streamer dataset, a large-scale co-speech gesture dataset emphasizing semantically explicit hand gestures in real-world streaming scenarios.

According to the paper appendix, the Streamer dataset includes:

- 18 semantic gesture categories
- 281 anchor actors
- about 58 hours of video
- 10-second clips sampled at 25 FPS
- 22 kHz audio

## Main Features

- Semantic-aware co-speech gesture generation with hybrid conditioning.
- Retrieval-augmented semantic gesture activation for more controllable outputs.
- Support for gesture editing settings including generation, gesture injection, in-betweening, and motion segment replacement.
- A repository structure designed for training, inference, evaluation, demos, and future checkpoint release.


## Data Preparation

The Streamer dataset is available at: [mumuwei/Streamer on Hugging Face](https://huggingface.co/datasets/mumuwei/Streamer)

The expected local dataset layout is:

```text
data/
`-- streamer-dataset
    ├── test_seen
    │   ├── anon_audios
    │   ├── audio_features
    │   └── gestures
    ├── test_unseen
    │   ├── anon_audios
    │   ├── audio_features
    │   └── gestures
    └── train
        ├── anon_audios
        ├── audio_features
        └── gestures
```
### 1. Convert PKL files to CPU

Raw gesture PKL files from SMPL-X optimization contain `losses_to_log` with CUDA
tensors. This step removes that field, converts any remaining GPU tensors to CPU
numpy, and overwrites the PKL files in-place.

```shell
python tools/convert_cpu.py
```

### 2. Generate CSV metadata

Scan `data/streamer-dataset/` and produce train / test_seen / test_unseen CSV files
together with `speaker_map.json` under `data/datasets/streamer/`:

```shell
python tools/prepare_csv.py
```

### 3. Extract audio features

Extract WavLM from raw wav
files and save as `.npy` under `data/streamer-dataset/{split}/audio_features/`:

```shell
python tools/generate_wavlm_feature.py --model_path ckpts/chinese-wav2vec2-large-fairseq-ckpt
```

### 4. Extract 3D keypoints (Optional for Training)

Run SMPL-X forward kinematics on gesture pkl files to produce 3D joint
positions under `data/streamer-dataset/{split}/keypoints_3d/`:

```shell
python tools/generate_keypoints_3d.py --body_model_path body_models
```


### 5. Generate WebDataset tar files (Optional for Training)

Pack gesture PKL, audio features, and 3D keypoints into WebDataset tar shards
for training under `data/streamer-dataset/{split}/tars/`:

```shell
python tools/generate_smplx_tar.py
```

This generates 30 tar shards for train, 10 for test_seen, and 10 for test_unseen
by default. Adjust with `--num_tars` and `--num_workers`.


## Rendering

Render SMPLX body motion from gesture PKL files into video. Requires OSMesa
for offscreen rendering.

```shell
# Render model predictions
bash render/render.sh \
    --pkl_file path/to/pkl_file\
    --smplx_model_path path/to/smplx_model_npz

# Render ground truth motion
bash render/render.sh --pkl_file path/to/gt.pkl --mode gt --save_path output.mp4

# Render with audio overlay
bash render/render.sh --pkl_file path/to/pred.pkl --mode ours --audio path/to/audio.wav --save_path output.mp4
```

Key options:
- `--pkl_file`: Input PKL file (ground truth or prediction). When `mode=ours`, the script will attempt to read `gt_path` from the PKL file automatically.
- `--mode`: `gt` for ground truth, `ours` for model predictions.
- `--smplx_model_path`: Path to the SMPLX model file (e.g. `SMPLX_MALE_shape2019_exp2020.npz`).
- `--gt_file`: Explicit path to the GT PKL file (optional, overrides `gt_path` in PKL).
- `--audio`: Audio file to overlay on the output video.
- `--save_path`: Output video path (default: `output.mp4`).


## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{yang2025gesturehydra,
  title   = {GestureHYDRA: Semantic Co-speech Gesture Synthesis via Hybrid Modality Diffusion Transformer and Cascaded-Synchronized Retrieval-Augmented Generation},
  author  = {Quanwei Yang and Luying Huang and Kaisiyuan Wang and Jiazhi Guan and Shengyi He and Fengguo Li and Hang Zhou and Lingyun Yu and Yingying Li and Haocheng Feng and Hongtao Xie},
  journal = {arXiv preprint arXiv:2507.22731},
  year    = {2025}
}
```

## Acknowledgement

This repository is currently in the initial release stage. Before making the project fully public, please update the final repository URL in `CITATION.cff` and add the final software and data license. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md) for the current release note.
