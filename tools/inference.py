"""Style-conditioned SMPL-X inference for the Streamer dataset."""

import argparse
import copy
import csv
import os
import pickle
import random

from gesturehydra.utils.warning_filters import configure_warning_filters

configure_warning_filters()
import librosa
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from scipy.ndimage import gaussian_filter
from torch.cuda.amp import autocast
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from gesturehydra.datasets.audio_smplx_dataset.rotation_conversion import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from gesturehydra.models import build_architecture


fix_index_3d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17,
                21, 22, 23, 24, 25, 26,
                30, 31, 32, 33, 34, 35,
                39, 40, 41, 42, 43, 44]
all_index_3d = np.ones(165)
all_index_3d[fix_index_3d] = 0

c_index_3d = []
i = 0
for num in all_index_3d:
    if num == 1:
        c_index_3d.append(i)
    i = i + 1
c_index_3d = np.asarray(c_index_3d)

c_index_6d = []
i = 0
for num in all_index_3d:
    if num == 1:
        c_index_6d.append(2 * i)
        c_index_6d.append(2 * i + 1)
    i = i + 1
c_index_6d = np.asarray(c_index_6d)


def motion_temporal_filter(motion, sigma=1):
    last_dim = motion.shape[-1]
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, last_dim)


AUDIO_SR = 16000
AUDIO_FPS = 25

# Lazy-loaded wav2vec2 model (initialized on first use)
_wav2vec_extractor = None
_wav2vec_model = None


def _ensure_wav2vec(model_path, device):
    """Load wav2vec2 model on first call, reuse afterwards."""
    global _wav2vec_extractor, _wav2vec_model
    if _wav2vec_model is None:
        print(f'Loading wav2vec2 model from {model_path} ...')
        _wav2vec_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        _wav2vec_model = Wav2Vec2Model.from_pretrained(model_path).to(device)
        _wav2vec_model.eval()
    return _wav2vec_extractor, _wav2vec_model


@torch.no_grad()
def load_audio(wav_path, model_path, device):
    """Load a wav file and extract wav2vec2 features.

    Returns:
        numpy array of shape (1, T, 1024), same format as pre-computed npy files.
    """
    feature_extractor, model = _ensure_wav2vec(model_path, device)

    wav, sr = librosa.load(wav_path, sr=AUDIO_SR, mono=False)
    if len(wav.shape) > 1:
        wav = wav[0]
    target_length = int(len(wav) / sr * AUDIO_FPS)

    # Chunked extraction (15s per chunk) to avoid OOM
    wav_tensor = torch.FloatTensor(wav).unsqueeze(0).to(device)
    input_values = feature_extractor(
        wav_tensor, sampling_rate=AUDIO_SR, return_tensors="pt",
    ).input_values.to(device)

    chunk_len = AUDIO_SR * 15
    wav_len = input_values.shape[-1]
    num_chunks = wav_len // chunk_len + 1
    input_values = F.pad(input_values, (0, chunk_len * num_chunks - wav_len))
    input_values = input_values.reshape(num_chunks, chunk_len)

    reps = []
    for i in range(0, num_chunks, 10):
        reps.append(model(input_values[i:i + 10]).last_hidden_state[0])
    rep = torch.cat(reps, dim=0)

    # Interpolate to target frame count
    feat = F.interpolate(
        rep.unsqueeze(0).transpose(1, 2),
        size=target_length,
        align_corners=True,
        mode='linear',
    ).transpose(1, 2).squeeze()  # (T, 1024)

    return feat.cpu().numpy()[np.newaxis]  # (1, T, 1024)


def parse_args():
    parser = argparse.ArgumentParser(description='gesturehydra style inference')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', type=str, required=True, help='output directory')
    parser.add_argument('--device',
                        choices=['cpu', 'cuda'],
                        default='cuda',
                        help='device used for testing')
    parser.add_argument('--seed-len', type=int, default=5,
                        help='number of seed frames for overlapping windows')
    parser.add_argument('--input-csv', type=str,
                        default='data/datasets/streamer/smplx_wav_va_test_seen.csv',
                        help='CSV file listing test samples')
    parser.add_argument('--style-csv', type=str,
                        default='data/datasets/streamer/smplx_wav_va_train.csv',
                        help='CSV file listing style reference samples')
    parser.add_argument('--wav2vec-model', type=str,
                        default='ckpts/chinese-wav2vec2-large-fairseq-ckpt',
                        help='Path to wav2vec2 model (used when pre-computed npy is missing)')
    args = parser.parse_args()
    return args


def get_origin_motion_seed(data, mean, std, motion_dim=258):
    jaw_pose = np.array(data['jaw_pose'])
    leye_pose = np.array(data['leye_pose'])
    reye_pose = np.array(data['reye_pose'])
    global_orient = np.array(data['global_orient']).squeeze()
    body_pose = np.array(data['body_pose_axis'])
    left_hand_pose = np.array(data['left_hand_pose'])
    right_hand_pose = np.array(data['right_hand_pose'])

    full_body = np.concatenate(
        (jaw_pose, leye_pose, reye_pose, global_orient, body_pose, left_hand_pose, right_hand_pose),
        axis=1)

    full_body = torch.from_numpy(full_body)
    full_body = matrix_to_rotation_6d(axis_angle_to_matrix(full_body.reshape(-1, 55, 3))).reshape(-1, 330)
    full_body = np.asarray(full_body)
    origin_motion = (full_body.copy() - mean) / std

    full_body_norm = (full_body - mean) / std
    full_body_norm = full_body_norm[:, c_index_6d]
    gesture = full_body_norm.copy()

    return origin_motion, gesture


def smooth_motion(pred_motion, mean, std, motion_dim=258):
    pred_motion = pred_motion[:, :motion_dim]
    origin_motion = np.zeros((pred_motion.shape[0], mean.shape[0]))

    result_body = origin_motion
    result_body[:, c_index_6d] = pred_motion

    final_6d_motion = np.multiply(result_body, std) + mean
    final_6d_motion_torch = torch.from_numpy(final_6d_motion).reshape(-1, 55, 6)
    final_3d_motion_torch = matrix_to_axis_angle(rotation_6d_to_matrix(final_6d_motion_torch.reshape(-1, 55, 6))).reshape(-1, 55, 3)

    body_pose_axis = motion_temporal_filter(final_3d_motion_torch[:, 4:25].cpu().numpy(), sigma=0.8)
    body_pose_axis = torch.from_numpy(body_pose_axis)
    final_3d_motion_torch[:, 4:25] = body_pose_axis

    left_hand_pose = motion_temporal_filter(final_3d_motion_torch[:, 25:40].cpu().numpy(), sigma=0.8)
    left_hand_pose = torch.from_numpy(left_hand_pose)
    final_3d_motion_torch[:, 25:40] = left_hand_pose

    right_hand_pose = motion_temporal_filter(final_3d_motion_torch[:, 40:].cpu().numpy(), sigma=0.8)
    right_hand_pose = torch.from_numpy(right_hand_pose)
    final_3d_motion_torch[:, 40:] = right_hand_pose

    full_body = matrix_to_rotation_6d(axis_angle_to_matrix(final_3d_motion_torch.reshape(-1, 55, 3))).reshape(-1, 330)
    full_body = np.asarray(full_body)

    full_body_norm = (full_body - mean) / std
    full_body_norm = full_body_norm[:, c_index_6d]
    gesture = full_body_norm.copy()

    return gesture


def format_result(pred_motion, origin_motion, origin_all_var, mean, std,
                  motion_dim=258, seed_len=25, with_seed=True):
    pred_motion = pred_motion[:, :motion_dim]
    if with_seed:
        motion_length = pred_motion.shape[0]
    else:
        motion_length = pred_motion.shape[0] - seed_len

    result_body = origin_motion
    result_body[:, c_index_6d] = pred_motion

    final_6d_motion = np.multiply(result_body, std) + mean
    final_6d_motion_torch = torch.from_numpy(final_6d_motion).reshape(-1, 55, 6).float()

    if with_seed:
        final_3d_motion_torch = matrix_to_axis_angle(rotation_6d_to_matrix(final_6d_motion_torch.reshape(-1, 55, 6))).reshape(-1, 55, 3)
    else:
        final_3d_motion_torch = matrix_to_axis_angle(rotation_6d_to_matrix(final_6d_motion_torch.reshape(-1, 55, 6))).reshape(-1, 55, 3)[seed_len:]

    origin_all_var['jaw_pose'] = final_3d_motion_torch[:, :1].reshape(motion_length, -1)
    origin_all_var['leye_pose'] = final_3d_motion_torch[:, 1:2].reshape(motion_length, -1)
    origin_all_var['reye_pose'] = final_3d_motion_torch[:, 2:3].reshape(motion_length, -1)
    origin_all_var['body_pose_axis'] = final_3d_motion_torch[:, 4:25].reshape(motion_length, -1)
    origin_all_var['left_hand_pose'] = final_3d_motion_torch[:, 25:40].reshape(motion_length, -1)
    origin_all_var['right_hand_pose'] = final_3d_motion_torch[:, 40:].reshape(motion_length, -1)
    origin_all_var['global_orient'] = final_3d_motion_torch[:, 3:4].reshape(motion_length, -1)

    expand_size = motion_length - len(origin_all_var['transl'])
    if expand_size > 0:
        origin_all_var['transl'] = np.concatenate([
            origin_all_var['transl'],
            origin_all_var['transl'][-1:, :].repeat(expand_size, axis=0)
        ], axis=0)
        origin_all_var['expression'] = np.concatenate([
            origin_all_var['expression'],
            origin_all_var['expression'][-1:, :].repeat(expand_size, axis=0)
        ], axis=0)
    else:
        origin_all_var['transl'] = origin_all_var['transl'][:motion_length]
        origin_all_var['expression'] = origin_all_var['expression'][:motion_length]

    for key in origin_all_var.keys():
        if isinstance(origin_all_var[key], torch.Tensor):
            origin_all_var[key] = origin_all_var[key].cpu().numpy()
    return origin_all_var


def main():
    torch.manual_seed(123456)

    args = parse_args()
    device = args.device
    save_path = args.out
    os.makedirs(save_path, exist_ok=True)

    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True
    motion_dim = cfg.data.test.motion_dim

    model = build_architecture(cfg.model)
    bf16_cfg = cfg.get('bf16', None)
    use_bf16 = bf16_cfg is not None
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.device == 'cpu':
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0])
    model.eval()
    model = model.to(device)

    max_length = cfg.data.test.sequence_length
    input_dim = motion_dim

    mean = np.zeros((330))
    std = np.ones((330))

    input_csv = args.input_csv
    style_csv = args.style_csv

    # Read test samples: CSV columns are [wav_path, pkl_path, speaker_name]
    datasets = []
    with open(input_csv, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            pkl_path = row[1]
            # npy_path = pkl_path.replace('/gestures/', '/audio_features/').replace('.pkl', '.npy')
            npy_path = pkl_path.replace('/gestures/', '/anon_audio_features/').replace('.pkl', '.npy')
            datasets.append({
                'wav_file': row[0],
                'pkl_file': pkl_path,
                'npy_file': npy_path,
                'speaker_name': row[2],
            })

    # Read style reference samples
    train_file_paths = []
    with open(style_csv, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            train_file_paths.append([row[1], row[2]])

    seed_len = args.seed_len

    for idx, file_data in enumerate(datasets):
        print(f'========= [{idx+1}/{len(datasets)}] pkl_file: {file_data["pkl_file"]}, '
              f'speaker: {file_data["speaker_name"]}')
        pkl_path = file_data['pkl_file']
        save_pkl_path = os.path.join(save_path, os.path.basename(os.path.dirname(os.path.dirname(pkl_path))),
                                     os.path.basename(os.path.dirname(pkl_path)),
                                     os.path.basename(pkl_path))
        if os.path.exists(save_pkl_path):
            print(f'========= skip: {save_pkl_path} already exists')
            continue

        with open(pkl_path, 'rb') as f:
            all_var = pickle.load(f)
        if isinstance(all_var, list):
            all_var = all_var[0]
        all_var = EasyDict(all_var)
        data = all_var
        betas = all_var['betas']

        try:
            audio_feat = np.load(file_data['npy_file'])
        except Exception:
            print(f'========= npy not found: {file_data["npy_file"]}, falling back to load_audio')
            try:
                audio_feat = load_audio(file_data['wav_file'], args.wav2vec_model, device)
            except Exception as e:
                print(f'========= skip: load_audio failed for {file_data["wav_file"]}: {e}')
                continue
        
        audio_feat = torch.from_numpy(audio_feat).to(device)

        speaker_name = file_data['speaker_name']

        betas = torch.from_numpy(betas).unsqueeze(0).to(device)

        origin_motion, origin_gesture = get_origin_motion_seed(data, mean, std, motion_dim)

        len_audio = audio_feat.shape[1]
        if len_audio < 180:
            print(f'========= skip: audio too short ({len_audio} < 180)')
            continue

        length_total_motion = int(len_audio)
        print(f'========= length_total_motion: {length_total_motion}')

        begin_seed_len = 1
        begin_seed = origin_gesture[0:0 + begin_seed_len]
        begin_seed = begin_seed.reshape(begin_seed.shape[0], -1)
        all_seed = torch.from_numpy(np.zeros([1, audio_feat.shape[1], input_dim])).to(device)
        all_seed[:, :begin_seed_len] = torch.from_numpy(begin_seed).unsqueeze(0).to(device)
        all_keyframe = torch.from_numpy(np.zeros([1, audio_feat.shape[1]])).to(device)
        all_keyframe[:, :begin_seed_len] = 1

        seed = all_seed[:, :max_length]
        key_frame = all_keyframe[:, :max_length]

        B = 4
        all_seed = all_seed.repeat(B, 1, 1)
        all_keyframe = all_keyframe.repeat(B, 1)
        seed = seed.repeat(B, 1, 1)
        key_frame = key_frame.repeat(B, 1)

        style_len = 75
        motion_style_candidates = [x[0] for x in train_file_paths
                                   if speaker_name == x[1] and pkl_path != x[0]]
        motion_style = []
        motion_style_mask = []
        for _ in range(B):
            while True:
                motion_style_file = random.choice(motion_style_candidates)
                with open(motion_style_file, 'rb') as file:
                    motion_style_var = pickle.load(file)
                motion_style_data = EasyDict(motion_style_var)
                motion_style_motion, motion_style_gesture = get_origin_motion_seed(
                    motion_style_data, mean, std, motion_dim)
                if len(motion_style_gesture) >= style_len:
                    break
            motion_style.append(torch.from_numpy(motion_style_gesture[:style_len]).unsqueeze(0).to(device))
            motion_style_mask.append(torch.from_numpy(np.ones(style_len)).unsqueeze(0).to(device))
        motion_style = torch.cat(motion_style, dim=0)
        motion_style_mask = torch.cat(motion_style_mask, dim=0)

        all_pred_motion = []
        frame_cnt = 0
        while frame_cnt < length_total_motion - seed_len:
            wavlm = audio_feat[:, frame_cnt:frame_cnt + max_length].to(device)
            seed[:, seed_len:] = all_seed[:, frame_cnt + seed_len:frame_cnt + max_length].to(device)
            key_frame[:, seed_len:] = all_keyframe[:, frame_cnt + seed_len:frame_cnt + max_length].to(device)
            motion = torch.zeros(1, max_length, input_dim).to(device)
            motion_mask = torch.ones(1, max_length).to(device)
            motion_length = torch.Tensor([max_length]).long().to(device)

            wavlm = wavlm.float()
            seed = seed.float()

            motion = motion.repeat(B, 1, 1)
            motion_mask = motion_mask.repeat(B, 1)
            motion_length = motion_length.repeat(B)
            wavlm = wavlm.repeat(B, 1, 1)
            betas = betas.repeat(B, 1, 1)
            input = {
                'motion': motion,
                'motion_mask': motion_mask,
                'motion_length': motion_length,
                'num_intervals': 1,
                'wavlm': wavlm,
                'seed': seed,
                'betas': betas,
                'motion_metas': [dict() for _ in range(B)],
                'key_frame': key_frame,
                'motion_style': motion_style,
                'motion_style_mask': motion_style_mask,
            }

            with torch.no_grad():
                if use_bf16:
                    with autocast(enabled=True, dtype=torch.bfloat16):
                        input['inference_kwargs'] = {}
                        output = model(**input)
                        pred_motion = [x['pred_motion'].cpu().detach().numpy() for x in output]
                        pred_motion = np.stack(pred_motion, axis=0)
                else:
                    input['inference_kwargs'] = {}
                    output = model(**input)
                    pred_motion = [x['pred_motion'].cpu().detach().numpy() for x in output]
                    pred_motion = np.stack(pred_motion, axis=0)

                pred_motion = np.clip(pred_motion, -1, 1)

                filter_pred_motion = []
                for bs in range(B):
                    filter_pred_motion.append(smooth_motion(pred_motion[bs], mean, std, motion_dim))
                pred_motion = np.stack(filter_pred_motion, axis=0)

                frame_cnt += max_length
                if not (frame_cnt == len_audio):
                    if frame_cnt + max_length > length_total_motion:
                        overlap_length = frame_cnt + max_length - length_total_motion
                        prev_length = frame_cnt - overlap_length + seed_len
                        pred_motion = pred_motion[:, :prev_length]
                        seed[:, :seed_len] = torch.from_numpy(pred_motion[:, -seed_len:]).to(device)
                        key_frame[:, :seed_len] = 1
                        frame_cnt = prev_length - seed_len
                    else:
                        seed[:, :seed_len] = torch.from_numpy(pred_motion[:, -seed_len:]).to(device)
                        key_frame[:, :seed_len] = 1
                        frame_cnt -= seed_len

            if len(all_pred_motion) > 0 and seed_len != 0:
                last_poses = all_pred_motion[-1][:, -seed_len:]
                all_pred_motion[-1] = all_pred_motion[-1][:, :-seed_len]
                for j in range(last_poses.shape[1]):
                    n = last_poses.shape[1]
                    prev = last_poses[:, j]
                    next_motion = pred_motion[:, j]
                    pred_motion[:, j] = prev * (n - j) / (n + 1) + next_motion * (j + 1) / (n + 1)

            all_pred_motion.append(pred_motion)

        all_preds = np.concatenate(all_pred_motion, axis=1)
        all_preds = all_preds[:, :len_audio]
        all_length = all_preds.shape[1]

        origin_motion = origin_motion[:all_length]
        all_smplx_results = []
        for bs in range(B):
            all_var_copy = copy.deepcopy(all_var)
            smplx_results = format_result(all_preds[bs], origin_motion, all_var_copy, mean, std, motion_dim, seed_len)
            all_smplx_results.append(copy.deepcopy(smplx_results))

        save_smplx_results = copy.deepcopy(all_smplx_results[0])
        save_smplx_results['jaw_pose'] = np.stack([x['jaw_pose'] for x in all_smplx_results], axis=0)
        save_smplx_results['leye_pose'] = np.stack([x['leye_pose'] for x in all_smplx_results], axis=0)
        save_smplx_results['reye_pose'] = np.stack([x['reye_pose'] for x in all_smplx_results], axis=0)
        save_smplx_results['body_pose_axis'] = np.stack([x['body_pose_axis'] for x in all_smplx_results], axis=0)
        save_smplx_results['left_hand_pose'] = np.stack([x['left_hand_pose'] for x in all_smplx_results], axis=0)
        save_smplx_results['right_hand_pose'] = np.stack([x['right_hand_pose'] for x in all_smplx_results], axis=0)
        save_smplx_results['global_orient'] = np.stack([x['global_orient'] for x in all_smplx_results], axis=0)
        save_smplx_results['transl'] = np.stack([x['transl'] for x in all_smplx_results], axis=0)
        save_smplx_results['expression'] = np.stack([x['expression'] for x in all_smplx_results], axis=0)
        save_smplx_results['gt_path'] = pkl_path

        os.makedirs(os.path.dirname(save_pkl_path), exist_ok=True)
        with open(save_pkl_path, 'wb') as file:
            pickle.dump(save_smplx_results, file)
        print(f'========= saved: {save_pkl_path}')


if __name__ == '__main__':
    main()
