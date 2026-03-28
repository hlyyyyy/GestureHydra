"""Generate WebDataset tar files from the streamer-dataset.

Reads CSV metadata (from tools/prepare_csv.py) and packs gesture PKL,
pre-extracted audio features (.npy) and 3D keypoints (.npy) into WebDataset
tar shards ready for training.

Output layout:
    data/streamer-dataset/{split}/tars/{000000..NNNNNN}.tar

Each tar sample contains:
    __key__          : unique sample identifier (relative pkl path without ext)
    poses.pickle     : zlib-compressed SMPL-X parameters dict
    wavlm.npy        : audio features, shape (1, T, feat_dim)
    keypoints_3d.npy : 3D joint positions, shape (T, 144, 3)
    speaker_name     : UTF-8 encoded speaker ID string (e.g. "000")

Usage:
    python tools/generate_smplx_tar.py \
        --dataset_root data/streamer-dataset \
        --csv_dir      data/datasets/streamer \
        --splits       train test_seen test_unseen \
        --num_tars     30 10 10 \
        --num_workers  24

Prerequisites:
    pip install webdataset easydict numpy tqdm
    Run tools/prepare_csv.py, tools/generate_wavlm_feature.py, and
    tools/generate_keypoints_3d.py first.
"""

import argparse
import csv
import multiprocessing as mp
import os
import pickle
import zlib
from io import BytesIO

import numpy as np
import webdataset as wds
from easydict import EasyDict
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Encode / decode helpers (same as original generate_tar.py)
# ---------------------------------------------------------------------------

def encode(obj, compress=True):
    """Serialize a python object via np.save and optionally zlib-compress."""
    with BytesIO() as out:
        np.save(out, obj)
        data = out.getvalue()
        if compress:
            data = zlib.compress(data, 1)
        return data


# ---------------------------------------------------------------------------
# Build sample list from CSV
# ---------------------------------------------------------------------------

def build_sample_list(csv_path, dataset_root):
    """Read a CSV file and return a list of sample dicts.

    Each dict contains absolute paths for pkl, audio feature npy, and
    keypoints_3d npy, plus the speaker_name string.
    """
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header: wav_path, pkl_path, speaker_name
        for row in reader:
            wav_path, pkl_path, speaker_name = row[0], row[1], row[2]

            # Derive audio_features path from wav_path:
            #   {split}/anon_audios/{anchor}/{md5}/{clip}.wav
            #   -> {split}/audio_features/{anchor}/{md5}/{clip}.npy
            npy_path = wav_path.replace("/anon_audios/", "/audio_features/")
            npy_path = os.path.splitext(npy_path)[0] + ".npy"

            # Derive keypoints_3d path from pkl_path:
            #   {split}/gestures/{anchor}/{md5}/{clip}.pkl
            #   -> {split}/keypoints_3d/{anchor}/{md5}/{clip}.npy
            kp_3d_path = pkl_path.replace("/gestures/", "/keypoints_3d/")
            kp_3d_path = os.path.splitext(kp_3d_path)[0] + ".npy"

            samples.append({
                "pkl_path": pkl_path,
                "npy_path": npy_path,
                "kp_3d_path": kp_3d_path,
                "speaker_name": speaker_name,
            })
    return samples


# ---------------------------------------------------------------------------
# Tar writing (multi-process)
# ---------------------------------------------------------------------------

def write_samples(dataset, tar_indices, sample_indices, save_dir):
    """Write a subset of tar shards (called per worker process)."""
    for t_idx, s_idx in zip(tar_indices, sample_indices):
        fname = os.path.join(save_dir, "%06d.tar" % t_idx)
        stream = wds.TarWriter(fname)
        for idx in tqdm(s_idx, desc=f"tar-{t_idx:06d}", position=t_idx % 4):
            try:
                entry = dataset[idx]
                pkl_path = entry["pkl_path"]
                npy_path = entry["npy_path"]
                kp_3d_path = entry["kp_3d_path"]
                speaker_name = entry["speaker_name"]

                # Load gesture pkl
                with open(pkl_path, "rb") as f:
                    all_var = pickle.load(f)
                if isinstance(all_var, list):
                    all_var = all_var[0]
                all_var = EasyDict(all_var)

                new_data = {}
                for key in all_var.keys():
                    if key == "losses_to_log":
                        continue
                    new_data[key] = all_var[key]

                # Load pre-extracted audio features
                wavlm = np.load(npy_path)
                # Load pre-extracted 3D keypoints
                kp_3d = np.load(kp_3d_path)

                # Build __key__ from pkl relative path (without extension)
                # e.g. "data/streamer-dataset/train/gestures/000/abc123/010_020"
                __key__ = os.path.splitext(pkl_path)[0]

                sample = {
                    "__key__": __key__,
                    "poses.pickle": encode(new_data, compress=True),
                    "wavlm.npy": wavlm,
                    "keypoints_3d.npy": kp_3d,
                    "speaker_name": speaker_name.encode("utf-8"),
                }
                stream.write(sample)
            except Exception as e:
                print(f"[ERROR] {entry.get('pkl_path', '?')}: {e}, skipping...")
        stream.close()


def dataset2tar(dataset, save_dir, num_tars, num_workers):
    """Distribute samples across tar shards and write in parallel."""
    os.makedirs(save_dir, exist_ok=True)

    num_len = len(dataset)
    data_index = list(range(num_len))
    # Interleave samples across shards (round-robin)
    samples = [data_index[i::num_tars] for i in range(num_tars)]
    tar_index = list(range(num_tars))

    actual_workers = min(num_workers, num_tars)
    jobs = []
    for i in range(actual_workers):
        job = mp.Process(
            target=write_samples,
            args=(dataset,
                  tar_index[i::actual_workers],
                  samples[i::actual_workers],
                  save_dir),
        )
        job.start()
        jobs.append(job)

    for job in jobs:
        job.join()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate WebDataset tar files from streamer-dataset."
    )
    parser.add_argument(
        "--dataset_root", type=str, default="data/streamer-dataset",
        help="Root of the streamer-dataset.",
    )
    parser.add_argument(
        "--csv_dir", type=str, default="data/datasets/streamer",
        help="Directory containing the CSV files from prepare_csv.py.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test_seen", "test_unseen"],
        help="Which splits to process.",
    )
    parser.add_argument(
        "--num_tars", nargs="+", type=int, default=[30, 10, 10],
        help="Number of tar shards per split (must match --splits length).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=24,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling samples.",
    )
    args = parser.parse_args()

    assert len(args.splits) == len(args.num_tars), \
        f"--splits ({len(args.splits)}) and --num_tars ({len(args.num_tars)}) must have the same length."

    for split, num_tars in zip(args.splits, args.num_tars):
        csv_path = os.path.join(args.csv_dir, f"smplx_wav_va_{split}.csv")
        if not os.path.exists(csv_path):
            print(f"[SKIP] CSV not found: {csv_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"  CSV:      {csv_path}")

        samples = build_sample_list(csv_path, args.dataset_root)
        print(f"  Samples:  {len(samples)}")

        # Shuffle samples
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(len(samples))
        samples = [samples[i] for i in perm]

        save_dir = os.path.join(args.dataset_root, split, "tars")
        print(f"  Output:   {save_dir}")
        print(f"  Shards:   {num_tars}")

        dataset2tar(samples, save_dir, num_tars=num_tars, num_workers=args.num_workers)

        print(f"  Done. Tar pattern: {save_dir}/{{000000..{num_tars-1:06d}}}.tar")


if __name__ == "__main__":
    main()
