import random
import numpy as np
from ..builder import PIPELINES

import os


@PIPELINES.register_module()
class LoadPreloadCropAudioCondition(object):
    def __init__(self, crop_size, audio_fps=25, audio_sr=16000, audio_aug=False, style_len=75):
        self.crop_size = crop_size
        self.audio_fps = audio_fps
        self.audio_sr = audio_sr
        self.audio_aug = audio_aug
        self.style_len = style_len

    def __call__(self, results):
        motion = results['motion']

        if not self.audio_aug:
            if 'wavlm' not in results:
                wavlm = np.load(results['wav_path'])
            else:
                wavlm = results['wavlm']
        else:
            if 'wavlm' not in results:
                random_index = random.randint(0, 4)
                wav_path = results['wav_path'].replace('.npy', f'_{random_index}.npy')
                while not os.path.exists(wav_path):
                    random_index = random.randint(0, 4)
                    wav_path = results['wav_path'].replace('.npy', f'_{random_index}.npy')
                wavlm = np.load(wav_path)
                results['wav_path'] = wav_path
            else:
                wavlm = results['wavlm']

        motion_length = len(motion)
        
        # Prevent length mismatch between wav and motion
        total_frame_len = min(len(wavlm), motion_length)
        motion = motion[:total_frame_len]
        wavlm = wavlm[:total_frame_len]
        if 'origin_motion' in results:
            results['origin_motion'] = results['origin_motion'][:total_frame_len]
        if 'expression' in results:
            results['expression'] = results['expression'][:total_frame_len]
        if 'trans' in results:
            results['trans'] = results['trans'][:total_frame_len]
        if 'gt_kp3d' in results:
            results['gt_kp3d'] = results['gt_kp3d'][:total_frame_len]
        
        if total_frame_len >= self.crop_size:
            idx = random.randint(0, total_frame_len - self.crop_size)
            # set test mode first idx to 0
            if results['test_mode']:
                idx = 0
            motion = motion[idx:idx + self.crop_size]
            wavlm = wavlm[idx:idx + self.crop_size]
            results['motion_length'] = self.crop_size
            results['motion_first_frame'] = idx
            if 'origin_motion' in results:
                results['origin_motion'] = results['origin_motion'][idx:idx + self.crop_size]
            if 'expression' in results:
                results['expression'] = results['expression'][idx:idx + self.crop_size]
            if 'trans' in results:
                results['trans'] = results['trans'][idx:idx + self.crop_size]
            if 'gt_kp3d' in results:
                results['gt_kp3d'] = results['gt_kp3d'][idx:idx + self.crop_size]
        else:
            padding_length = self.crop_size - total_frame_len
            D = motion.shape[1:]
            D_wavlm = wavlm.shape[1:]
            padding_zeros_motion = np.zeros((padding_length, *D), dtype=np.float32)
            padding_zeros_wavlm = np.zeros((padding_length, *D_wavlm), dtype=np.float32)
            motion = np.concatenate([motion, padding_zeros_motion], axis=0)
            wavlm = np.concatenate([wavlm, padding_zeros_wavlm], axis=0)
            if 'origin_motion' in results:
                D_origin_motion = results['origin_motion'].shape[1:]
                padding_zeros_origin_motion = np.zeros((padding_length, *D_origin_motion), dtype=np.float32)
                results['origin_motion'] = np.concatenate([results['origin_motion'], padding_zeros_origin_motion], axis=0)
            if 'expression' in results:
                D_expression = results['expression'].shape[1:]
                padding_zeros_expression = np.zeros((padding_length, *D_expression), dtype=np.float32)
                results['expression'] = np.concatenate([results['expression'], padding_zeros_expression], axis=0)
            if 'trans' in results:
                D_trans = results['trans'].shape[1:]
                padding_zeros_trans = np.zeros((padding_length, *D_trans), dtype=np.float32)
                results['trans'] = np.concatenate([results['trans'], padding_zeros_trans], axis=0)
            if 'gt_kp3d' in results:
                D_gt_kp3d = results['gt_kp3d'].shape[1:]
                padding_zeros_gt_kp3d = np.zeros((padding_length, *D_gt_kp3d), dtype=np.float32)
                results['gt_kp3d'] = np.concatenate([results['gt_kp3d'], padding_zeros_gt_kp3d], axis=0)
            results['motion_length'] = total_frame_len
            results['motion_first_frame'] = 0
        
        # Process motion_style
        if "motion_style" in results:
            motion_style = results['motion_style']
            if len(motion_style) < self.style_len:
                padding_length_style = self.style_len - len(motion_style)
                D_style = motion_style.shape[1:]
                padding_zeros_motion_style = np.zeros((padding_length_style, *D_style), dtype=np.float32)
                motion_style = np.concatenate([motion_style, padding_zeros_motion_style], axis=0)
                motion_style_mask = np.concatenate(
                    (np.ones(len(motion_style)),
                    np.zeros(self.style_len - len(motion_style))))
            else:
                style_idx = random.randint(0, len(motion_style) - self.style_len)
                motion_style = motion_style[style_idx:style_idx + self.style_len]
                motion_style_mask = np.ones(self.style_len)
            results['motion_style'] = motion_style
            results['motion_style_mask'] = motion_style_mask
        
        assert len(motion) == self.crop_size
        results['motion'] = motion
        results['wavlm'] = wavlm
        results['motion_shape'] = motion.shape

        if total_frame_len >= self.crop_size:
            results['motion_mask'] = np.ones(self.crop_size)
        else:
            results['motion_mask'] = np.concatenate(
                (np.ones(total_frame_len),
                 np.zeros(self.crop_size - total_frame_len)))
        
        return results


@PIPELINES.register_module()
class GetSeedConditionWithKeyFrame(object):
    def __init__(
        self, 
        short_seg_num=3,
        short_window=[10, 25],
        long_window=[80, 120],
        strategy=None,
    ):
        # Randomly choose from 3 strategies:
        # 1. Randomly mask a long segment of motion
        # 2. Randomly mask 50%~70% of frames
        # 3. Randomly provide GT for multiple consecutive motion segments
        self.short_seg_num = short_seg_num
        self.short_window = short_window
        self.long_window = long_window
        self.strategy = strategy

    def find_positive_segments(self, arr):
        segments = []
        start = None
        
        for i in range(len(arr)):
            # Found a positive number and no segment recording has started
            if arr[i] > 0 and start is None:
                start = i
            # Current number is 0 or reached the end of array, and a segment was being recorded
            elif (arr[i] <= 0 or i == len(arr)-1) and start is not None:
                # If reached the end of array and the last number is positive
                if i == len(arr)-1 and arr[i] > 0:
                    segments.append((start, i))
                else:
                    segments.append((start, i-1))
                start = None
                
        return segments
    
    def __call__(self, results):
        if self.strategy is None:
            strategy = random.randint(0, 3)
        else:
            strategy = self.strategy

        motion = results['motion']
        motion_length = results['motion_length']

        test_mode = results['test_mode']

        if not test_mode:
            if strategy == 0:
                # Strategy 1: Randomly mask a long segment
                window_size = random.randint(self.long_window[0], min(self.long_window[1], motion_length))
                seed = motion.copy()
                # initialize key frame arrays
                key_frame = np.ones((motion.shape[0]))
                start = random.randint(0, motion_length - window_size)
                key_frame[start:start + window_size] = 0
                seed[start:start + window_size] = 0

            elif strategy == 1:
                # Strategy 2: Randomly mask 50-70% frames
                seed = motion.copy()
                # initialize key frame arrays
                key_frame = np.ones((motion.shape[0]))
                # Calculate number of frames to mask based on random ratio between 50-70%
                mask_ratio = random.uniform(0.6, 0.9)
                num_frames_to_mask = int(motion_length * mask_ratio)
                
                # Create array of indices and randomly select which ones to mask
                indices = np.arange(motion_length)
                mask_indices = np.random.choice(indices, size=num_frames_to_mask, replace=False)
                mask_indices = [x for x in mask_indices if x >= 5]
                
                # Set selected indices to 0 in both seed and key_frame
                seed[mask_indices] = 0
                key_frame[mask_indices] = 0

            elif strategy == 2:
                # Strategy 3: Random multiple short segments
                # First randomly determine how many segments to select
                num_selected_segments = random.randint(1, self.short_seg_num)
                seed = np.zeros_like(motion)
                key_frame = np.zeros((motion.shape[0]))
                for _ in range(num_selected_segments):
                    window_size = random.randint(self.short_window[0], min(self.short_window[1], motion_length))
                    start_indices = random.randint(0, motion_length - window_size)
                    seed[start_indices:start_indices + window_size] = motion[start_indices:start_indices + window_size]
                    key_frame[start_indices:start_indices + window_size] = 1

            elif strategy == 3:
                # Strategy 3: Just give start frames
                seed = np.zeros_like(motion)
                key_frame = np.zeros((motion.shape[0]))
                num_start_frames = random.randint(1, 10)
                seed[:num_start_frames] = motion[:num_start_frames]
                key_frame[:num_start_frames] = 1

        else:
            seed = np.zeros_like(motion)
            key_frame = np.zeros((motion.shape[0]))
            seed[:5] = motion[:5]
            key_frame[:5] = 1

        results['seed'] = seed
        results['key_frame'] = key_frame
        
        return results
