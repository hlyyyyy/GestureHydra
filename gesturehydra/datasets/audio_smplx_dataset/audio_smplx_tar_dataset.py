import webdataset as wds
import numpy as np
import random
import torch
from io import BytesIO
import pickle
from easydict import EasyDict
import zlib
from gesturehydra.datasets.pipelines import Compose
from gesturehydra.datasets.builder import DATASETS
from gesturehydra.datasets.audio_smplx_dataset.rotation_conversion import axis_angle_to_matrix, matrix_to_rotation_6d
from mmcv.parallel import DataContainer


""" get trainable idx """
fix_index_3d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17,
                21, 22, 23, 24, 25, 26,
                30, 31, 32, 33, 34, 35,
                39, 40, 41, 42, 43, 44]  # 11x3 without global_orient
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


def decode(data, decompress=True):
    """data: bytes"""
    if decompress:
        data = zlib.decompress(data)
    np_data = np.load(BytesIO(data), allow_pickle=True).item()
    return np_data


""" Training Dataloader """
@DATASETS.register_module()
class SmplxA2GPreLoadDataloader(object):
    NAME_DICT = {f'{i:03d}': i for i in range(281)}
    def __init__(
        self,
        data_path,
        pipeline,
        motion_dim,
        sequence_length,
        dataset_name,
        csv_data_path=None,
        batch_size=64,
        num_workers=4,
        resampled=True,
        min_length=75,
        use_expression=False,
        initialize_smplx_model=False,
        use_motion_style=False,
        test_mode=False,
        **kwargs,
    ):
        self.data_path = data_path
        self.pipeline = Compose(pipeline)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resampled = resampled
        self.sequence_length = sequence_length
        self.use_expression = use_expression
        self.min_length = min_length
        self.initialize_smplx_model = initialize_smplx_model
        self.test_mode = test_mode
        self.dataset_name = dataset_name
        self.use_motion_style = use_motion_style
        self.csv_data_path = csv_data_path

        if self.csv_data_path is not None:
            # Read contents from csv
            import csv
            train_file_paths = []
            with open(self.csv_data_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                # Skip header
                next(reader)
                # Iterate over each row
                for row in reader:
                    train_file_paths.append([row[1], row[2]])  # pkl, speakername
            self.train_file_paths = train_file_paths

    def decode_pose(self, data):
        jaw_pose = np.array(data['jaw_pose'])     # (253, 3) 0
        leye_pose = np.array(data['leye_pose'])   # (253, 3) 1
        reye_pose = np.array(data['reye_pose'])   # (253, 3) 2
        global_orient = np.array(data['global_orient']).squeeze()  # (253, 3) 3
        body_pose = np.array(data['body_pose_axis'])  # (253, 63) 4~24 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 free
        left_hand_pose = np.array(data['left_hand_pose'])  # (253, 45) 25~39
        right_hand_pose = np.array(data['right_hand_pose'])  # (253, 45) 40~54
        expression = np.array(data['expression'])  # (253, 50)
        betas = np.array(data['betas'])
        trans = np.array(data['transl'])

        full_body = np.concatenate(
            (jaw_pose, leye_pose, reye_pose, global_orient, body_pose, left_hand_pose, right_hand_pose), axis=1) # (n_seq, 165)
        origin_motion = full_body.copy()

        # convert_to_6d
        full_body = torch.from_numpy(full_body)  # torch.Size([250, 165])
        full_body = matrix_to_rotation_6d(axis_angle_to_matrix(full_body.reshape(-1, 55, 3))).reshape(-1, 330) # torch.Size([n_seq, 330])
        full_body = np.asarray(full_body)

        full_body = full_body[:, c_index_6d]

        gesture = full_body.copy()

        # expression
        if self.use_expression:
            gesture = np.concatenate((gesture, expression), axis=-1)

        return gesture, origin_motion, expression, betas, trans

    def get_data_info(self, sample):
        data = decode(sample['poses.pickle'], decompress=True)
        wavlm = np.load(BytesIO(sample['wavlm.npy']), allow_pickle=True)[0]

        if self.initialize_smplx_model:
            kp_3d = np.load(BytesIO(sample['keypoints_3d.npy']), allow_pickle=True)
        if 'speaker_name' in sample:
            speaker_name = sample['speaker_name'].decode()
        else:
            speaker_name = None

        # Ensure data length is at least 3s (seed=1s, completion task seed=1s, generate at least 1s)
        if len(data['jaw_pose']) < self.min_length:
            return None
        gesture, origin_motion, expression, betas, trans = self.decode_pose(data)

        if self.use_motion_style:
            __key__ = sample['__key__']
            motion_style_candidates = [x[0] for x in self.train_file_paths if speaker_name == x[1] and __key__ != x[0]]
            motion_style_file = random.choice(motion_style_candidates)
            with open(motion_style_file, 'rb') as file:
                motion_style_var = pickle.load(file)
            if isinstance(motion_style_var, list):
                motion_style_var = motion_style_var[0]
            motion_style_data = EasyDict(motion_style_var)
            motion_style_gesture, _, _, _, _ = self.decode_pose(motion_style_data)

        # Organize the output as a dict
        results = {
            'motion': gesture,  # (n_seq, 774)  6d motion + vel + acc
            'wavlm': wavlm,
            'origin_motion': origin_motion,
            'expression': expression,
            'betas': betas,
            'trans': trans,
            'test_mode': self.test_mode,
        }
        results['dataset_name'] = self.dataset_name
        if self.initialize_smplx_model:
            results['gt_kp3d'] = kp_3d
        if self.use_motion_style:
            results['motion_style'] = motion_style_gesture
        return results

    def get_clip(self, sample):
        try:
            results = self.get_data_info(sample)
            if results is None:
                return None
            results = self.pipeline(results)
            return results
        except Exception as e:
            print(f"========================= Skip sample: {sample['__key__']} {e}")
            return None
    
    def custom_collate_fn(self, batch):
        """
        Custom collate function to handle dictionary batches
        """
        if len(batch) == 0:
            return {}
        
        collated_batch = {}
        # Get all keys from the first item
        keys = batch[0].keys()
        
        for key in keys:
            if isinstance(batch[0][key], np.ndarray):
                # Stack numpy arrays
                collated_batch[key] = np.stack([item[key] for item in batch], axis=0)
            elif isinstance(batch[0][key], torch.Tensor):
                collated_batch[key] = torch.stack([item[key] for item in batch], axis=0)
            # elif isinstance(batch[0][key], dict):
            #     # Handle nested dictionaries (like motion_metas)
            #     collated_batch[key] = {}
            #     nested_keys = batch[0][key].keys()
            #     for nested_key in nested_keys:
            #         collated_batch[key][nested_key] = [item[key][nested_key] for item in batch]
            elif isinstance(batch[0][key], DataContainer):
                collated_batch[key] = [item[key].data for item in batch]
            else:
                # Handle other types (like strings, integers, etc.)
                collated_batch[key] = [item[key] for item in batch]
        return collated_batch
    
    def get_clip_wrapper(self, x):
        result = self.get_clip(x)
        return result
    
    def make_loader(self):
        dataset = (
            wds.WebDataset(
                self.data_path,
                repeat=True,
                cache_dir='/tmp',
                shardshuffle=1000,
                resampled=self.resampled,
                handler=wds.warn_and_continue,
                nodesplitter=None
            )
            .shuffle(2000)
            .map(self.get_clip_wrapper)
            .batched(self.batch_size, collation_fn=self.custom_collate_fn, partial=False)
        )
        # collation_fn=self.custom_collate_fn
        loader = wds.WebLoader(
            dataset, 
            batch_size=None, 
            shuffle=False,
            num_workers=self.num_workers,
        )
        return loader

