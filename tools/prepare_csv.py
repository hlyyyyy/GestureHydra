"""Generate CSV metadata files from the streamer-dataset.

This script scans `data/streamer-dataset/{train,test_seen,test_unseen}`
and produces CSV files consumed by the training / inference pipeline:

    data/datasets/streamer/smplx_wav_va_train.csv
    data/datasets/streamer/smplx_wav_va_test_seen.csv
    data/datasets/streamer/smplx_wav_va_test_unseen.csv

Each CSV has three columns:  wav_path, pkl_path, speaker_name

Usage:
    python tools/prepare_csv.py [--dataset_root data/streamer-dataset]
                                [--output_dir  data/datasets/streamer]
"""

import argparse
import csv
import json
import os
from pathlib import Path


SPLITS = {
    "train": "smplx_wav_va_train.csv",
    "test_seen": "smplx_wav_va_test_seen.csv",
    "test_unseen": "smplx_wav_va_test_unseen.csv",
}


def collect_pairs(split_dir: Path):
    """Collect (wav_path, pkl_path, speaker_name) tuples from a split dir.

    Expected layout:
        split_dir/anon_audios/{anchor_id}/{video_md5}/{start}_{end}.wav
        split_dir/gestures/{anchor_id}/{video_md5}/{start}_{end}.pkl
    """
    audio_dir = split_dir / "anon_audios"
    gesture_dir = split_dir / "gestures"

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not gesture_dir.exists():
        raise FileNotFoundError(f"Gesture directory not found: {gesture_dir}")

    rows = []
    for anchor_id in sorted(os.listdir(audio_dir)):
        anchor_audio = audio_dir / anchor_id
        if not anchor_audio.is_dir():
            continue
        for video_md5 in sorted(os.listdir(anchor_audio)):
            video_audio = anchor_audio / video_md5
            if not video_audio.is_dir():
                continue
            for wav_file in sorted(os.listdir(video_audio)):
                if not wav_file.endswith(".wav"):
                    continue
                clip_name = wav_file[: -len(".wav")]  # e.g. "000_010"
                pkl_file = clip_name + ".pkl"

                wav_path = video_audio / wav_file
                pkl_path = gesture_dir / anchor_id / video_md5 / pkl_file

                if not pkl_path.exists():
                    print(f"[WARN] Missing gesture file: {pkl_path}")
                    continue

                rows.append(
                    (str(wav_path), str(pkl_path), anchor_id)
                )
    return rows


def build_speaker_map(all_rows):
    """Build a speaker_name -> int mapping from all collected rows."""
    speakers = sorted(set(r[2] for r in all_rows))
    return {name: idx for idx, name in enumerate(speakers)}


def main():
    parser = argparse.ArgumentParser(description="Generate CSV metadata from streamer-dataset.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/streamer-dataset",
        help="Root directory of the streamer-dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/datasets/streamer",
        help="Directory to write the CSV files.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []  # accumulate rows from all splits for speaker_map

    for split_name, csv_name in SPLITS.items():
        split_dir = dataset_root / split_name
        if not split_dir.exists():
            print(f"[SKIP] Split directory not found: {split_dir}")
            continue

        rows = collect_pairs(split_dir)
        all_rows.extend(rows)

        csv_path = output_dir / csv_name
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wav_path", "pkl_path", "speaker_name"])
            writer.writerows(rows)

        print(f"[OK] {csv_name}: {len(rows)} samples")

    # Write speaker_map.json
    speaker_map = build_speaker_map(all_rows)
    speaker_map_path = output_dir / "speaker_map.json"
    with open(speaker_map_path, "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, indent=2, ensure_ascii=False)
    print(f"[OK] speaker_map.json: {len(speaker_map)} speakers")


if __name__ == "__main__":
    main()
