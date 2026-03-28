"""Extract WavLM audio features from raw .wav files in the streamer-dataset
and save as .npy files.

Output layout (parallel to anon_audios/ and gestures/):
    data/streamer-dataset/{split}/audio_features/{anchor_id}/{video_md5}/{start}_{end}.npy

Each .npy file has shape (T, 1024) containing WavLM features.

Usage:
    python tools/generate_wavlm_feature.py \
        --dataset_root data/streamer-dataset \
        --model_path   ckpts/chinese-wav2vec2-large-fairseq-ckpt \
        --splits       train test_seen test_unseen \
        --device       cuda

Prerequisites:
    pip install transformers librosa
    The chinese-wav2vec2-large model checkpoint must be available at --model_path.
"""

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

AUDIO_SR = 16000
AUDIO_FPS = 25


# ─── WavLM feature extraction ────────────────────────────────────────────
def build_wav2vec(model_path: str, device: torch.device):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2Model.from_pretrained(model_path).to(device)
    model.eval()
    return feature_extractor, model


@torch.no_grad()
def wav2feat_cn(wav_input_16khz, feature_extractor, model, device):
    input_values = feature_extractor(
        wav_input_16khz.to(device),
        sampling_rate=AUDIO_SR,
        return_tensors="pt",
    ).input_values.to(device)

    chunk_len = AUDIO_SR * 15  # 15 seconds per chunk
    wav_len = input_values.shape[-1]
    num_chunks = wav_len // chunk_len + 1
    input_values = F.pad(input_values, (0, chunk_len * num_chunks - wav_len))
    input_values = input_values.reshape(num_chunks, chunk_len)

    reps = []
    for i in range(0, num_chunks, 10):
        reps.append(model(input_values[i : i + 10]).last_hidden_state[0])
    rep = torch.cat(reps, dim=0)
    del input_values
    return rep


# ─── per-file feature extraction ─────────────────────────────────────────
def extract_and_save(wav_path: str, save_path: str,
                     feature_extractor, model, device):
    """Extract WavLM features from a single wav file and save as npy."""
    wav, sr = librosa.load(wav_path, sr=AUDIO_SR, mono=False)
    if len(wav.shape) > 1:
        wav = wav[0]

    target_length = int(len(wav) / sr * AUDIO_FPS)

    # WavLM features
    wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
    wavlm_f = wav2feat_cn(wav_tensor, feature_extractor, model, device)
    wavlm_f = (
        F.interpolate(
            wavlm_f.unsqueeze(0).transpose(1, 2),
            size=target_length,
            align_corners=True,
            mode="linear",
        )
        .transpose(1, 2)
        .squeeze()
    )  # (T, 1024)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, wavlm_f.cpu().numpy()[np.newaxis])  # (1, T, 1024)


# ─── main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract audio features from streamer-dataset wav files."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/streamer-dataset",
        help="Root of the streamer-dataset.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="ckpts/chinese-wav2vec2-large-fairseq-ckpt",
        help="Path to the chinese-wav2vec2 model checkpoint.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test_seen", "test_unseen"],
        help="Which splits to process.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for WavLM inference (cuda or cpu).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    device = torch.device(args.device)

    print(f"Loading wav2vec2 model from {args.model_path} ...")
    feature_extractor, model = build_wav2vec(args.model_path, device)

    for split in args.splits:
        split_dir = dataset_root / split
        audio_dir = split_dir / "anon_audios"
        feature_dir = split_dir / "audio_features"

        if not audio_dir.exists():
            print(f"[SKIP] {audio_dir} not found")
            continue

        # Collect all wav files
        wav_files = sorted(audio_dir.rglob("*.wav"))
        print(f"\n[{split}] Found {len(wav_files)} wav files")

        for wav_path in tqdm(wav_files, desc=split):
            # anon_audios/{anchor}/{md5}/{clip}.wav -> audio_features/{anchor}/{md5}/{clip}.npy
            rel = wav_path.relative_to(audio_dir)
            save_path = feature_dir / rel.with_suffix(".npy")

            if save_path.exists():
                continue

            try:
                extract_and_save(
                    str(wav_path), str(save_path),
                    feature_extractor, model, device,
                )
            except Exception as e:
                print(f"[ERROR] {wav_path}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
