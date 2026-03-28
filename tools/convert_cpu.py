import pickle
import torch
import numpy as np
from easydict import EasyDict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import csv
import argparse


def convert_to_cpu(pkl_file):
    """Load a pkl file, convert any GPU tensors to numpy, remove losses_to_log, and overwrite in-place."""
    with open(pkl_file, 'rb') as f:
        all_var = pickle.load(f)
    if isinstance(all_var, list):
        all_var = all_var[0]
    all_var = EasyDict(all_var)

    new_data = {}
    for key in all_var.keys():
        if key == 'losses_to_log':
            continue
        if isinstance(all_var[key], torch.Tensor):
            new_data[key] = all_var[key].cpu().numpy()
        else:
            new_data[key] = all_var[key]
    new_data = EasyDict(new_data)

    with open(pkl_file, 'wb') as f:
        pickle.dump(new_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert GPU tensors in pkl files to CPU numpy and remove losses_to_log.')
    parser.add_argument(
        '--csv_paths', nargs='+',
        default=[
            'data/datasets/streamer/smplx_wav_va_train.csv',
            'data/datasets/streamer/smplx_wav_va_test_seen.csv',
            'data/datasets/streamer/smplx_wav_va_test_unseen.csv',
        ],
        help='CSV files containing pkl_path column (default: streamer train/test_seen/test_unseen)')
    parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()

    # Collect all pkl paths from CSV files
    all_pkl_path_list = []
    for csv_path in args.csv_paths:
        if not os.path.exists(csv_path):
            print(f'WARNING: {csv_path} not found, skipping')
            continue
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            for row in reader:
                pkl_path = row[1]
                if os.path.exists(pkl_path):
                    all_pkl_path_list.append(pkl_path)
                else:
                    print(f'WARNING: {pkl_path} not found, skipping')

    print(f'Total pkl files to process: {len(all_pkl_path_list)}')

    task_pool = set()
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        pbar = tqdm(total=len(all_pkl_path_list), desc='submit tasks')
        for pkl_path in all_pkl_path_list:
            task_pool.add(executor.submit(convert_to_cpu, pkl_path))
            pbar.update(1)
        pbar.close()

        pbar = tqdm(total=len(task_pool), desc='convert to cpu')
        for task in as_completed(task_pool):
            pbar.update(1)
            try:
                task.result()
            except Exception as e:
                print(f'ERROR: {e}')
        pbar.close()

    print('Done.')
