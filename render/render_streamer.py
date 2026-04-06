"""Render SMPLX body motion from pkl files.

This script provides visualization tools for the dataset. It supports
rendering ground truth (GT) motion data and model predictions from pkl files
into video using SMPLX body model and pyrender-based offscreen rendering.

Usage:
    python render_streamer.py --pkl_file path/to/data.pkl --save_path output.mp4
    python render_streamer.py --pkl_file path/to/data.pkl --audio path/to/audio.wav --save_path output.mp4
    python render_streamer.py --pkl_file path/to/data.pkl --mode ours --save_path output.mp4
"""

import argparse
import json
import os
import pickle

from gesturehydra.utils.warning_filters import configure_warning_filters

configure_warning_filters()
import mmcv
import numpy as np
import smplx
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

from rendering import RenderTool
from rotation_conversion import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

# Directory of this script, used for resolving relative paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default fixed global orientation for lower body
FIXED_GLOBAL_ORIENT = torch.tensor([3.0747, -0.0158, -0.0152])

# Default lower body pose (used to fill fixed joints)
LOWER_POSE = torch.tensor([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    3.0747, -0.0158, -0.0152,
    -1.1826512813568115, 0.23866955935955048, 0.15146760642528534,
    -1.2604516744613647, -0.3160211145877838, -0.1603458970785141,
    1.1654603481292725, 0.0, 0.0,
    1.2521806955337524, 0.041598282754421234, -0.06312154978513718,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
])

# Number of expression coefficients
EXP_DIM = 100

# Load hand PCA components
_hand_component_path = os.path.join(_SCRIPT_DIR, 'hand_component.json')
with open(_hand_component_path) as f:
    _comp = json.load(f)
    LEFT_HAND_COMPONENTS = np.asarray(_comp['left'])
    RIGHT_HAND_COMPONENTS = np.asarray(_comp['right'])

# Fixed joint indices in body_pose for lower body
FIXED_BODY_INDICES = [
    0, 1, 2, 3, 4, 5,
    9, 10, 11, 12, 13, 14,
    18, 19, 20, 21, 22, 23,
]

def as_numpy_float32(value):
    return np.asarray(value, dtype=np.float32).copy()


def patch_smplx_create_mean_pose():
    """Patch smplx mean-pose creation for older open-source releases."""

    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _create_mean_pose(self, data_struct, flat_hand_mean=False):
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        body_pose_mean = torch.zeros([self.NUM_BODY_JOINTS * 3], dtype=self.dtype)
        jaw_pose_mean = torch.zeros([3], dtype=self.dtype)
        leye_pose_mean = torch.zeros([3], dtype=self.dtype)
        reye_pose_mean = torch.zeros([3], dtype=self.dtype)

        pose_mean = np.concatenate([
            _to_numpy(global_orient_mean),
            _to_numpy(body_pose_mean),
            _to_numpy(jaw_pose_mean),
            _to_numpy(leye_pose_mean),
            _to_numpy(reye_pose_mean),
            _to_numpy(self.left_hand_mean),
            _to_numpy(self.right_hand_mean),
        ], axis=0)
        return pose_mean

    smplx.body_models.SMPLX.create_mean_pose = _create_mean_pose



def to3d_local(data):
    """Convert 12-dim PCA hand pose to 45-dim full hand pose via components."""
    left_hand_pose = np.einsum(
        'bi,ij->bj', data[:, 75:87], LEFT_HAND_COMPONENTS[:12, :]
    )
    right_hand_pose = np.einsum(
        'bi,ij->bj', data[:, 87:99], RIGHT_HAND_COMPONENTS[:12, :]
    )
    data = np.concatenate(
        (data[:, :75], left_hand_pose, right_hand_pose), axis=-1
    )
    return data


def pred_6d_to_3d(poses, is_expression=True):
    """Convert 6D rotation representation to axis-angle (3D).

    Args:
        poses: Tensor of shape (B, T, C) where the last EXP_DIM dims are
            expression coefficients (if is_expression=True).
        is_expression: Whether the last EXP_DIM dims are expressions.

    Returns:
        Tensor of shape (B, T, C') in axis-angle format.
    """
    bs, t, _ = poses.shape
    poses_exp = poses[..., -EXP_DIM:] if is_expression else None
    poses = poses[..., :-EXP_DIM] if is_expression else poses

    poses = poses.reshape(-1, 6)
    poses = matrix_to_axis_angle(rotation_6d_to_matrix(poses)).reshape(bs, t, -1)

    if is_expression:
        poses = torch.cat([poses, poses_exp], -1)
    return poses


def part2full(input_pose, stand=False):
    """Expand partial body pose to full body pose by filling fixed joints.

    Args:
        input_pose: Tensor of shape (T, C_partial).
        stand: If True, use standing pose for lower body.

    Returns:
        Tensor of shape (T, C_full) with fixed joints filled.
    """
    if stand:
        lp = torch.zeros_like(LOWER_POSE)
        lp[6:9] = torch.tensor([3.0747, -0.0158, -0.0152])
        lp = lp.unsqueeze(0).repeat(input_pose.shape[0], 1).to(input_pose.device)
    else:
        lp = LOWER_POSE.unsqueeze(0).repeat(input_pose.shape[0], 1).to(input_pose.device)

    output = torch.cat([
        input_pose[:, :3],   # jaw pose
        lp[:, :15],          # fixed: leye, reye, global_orient, joint1, joint2
        input_pose[:, 3:6],  # joint3 (predicted)
        lp[:, 15:21],        # fixed: joint4, joint5
        input_pose[:, 6:9],  # joint6 (predicted)
        lp[:, 21:27],        # fixed: joint7, joint8
        input_pose[:, 9:],   # remaining joints
    ], dim=1)
    return output


def poses2pred_local(input_pose, stand=False, fix_global=True):
    """Fill fixed lower-body joints into a full pose sequence.

    Args:
        input_pose: Tensor of shape (T, C) full-body pose.
        stand: If True, use standing lower body pose.
        fix_global: If True, use default global orientation.

    Returns:
        Tensor of shape (T, C) with fixed joints overwritten.
    """
    if stand:
        lp = LOWER_POSE.unsqueeze(0).repeat(input_pose.shape[0], 1).to(input_pose.device)
    else:
        lp = LOWER_POSE.unsqueeze(0).repeat(input_pose.shape[0], 1).to(input_pose.device)

    output = torch.cat([
        input_pose[:, :3],     # jaw pose (GT)
        lp[:, :15],            # fixed: leye, reye, global_orient, joint1, joint2
        input_pose[:, 18:21],  # joint3 (GT)
        lp[:, 15:21],          # fixed: joint4, joint5
        input_pose[:, 27:30],  # joint6 (GT)
        lp[:, 21:27],          # fixed: joint7, joint8
        input_pose[:, 36:],    # remaining joints (GT)
    ], dim=1)
    if not fix_global:
        output[:, 9:12] = input_pose[:, 9:12]
    return output


def get_smplx_to_pyrender_K(cam_transl):
    """Create a 4x4 extrinsic matrix from SMPLX camera translation."""
    if isinstance(cam_transl, torch.Tensor):
        T = cam_transl.detach().cpu().numpy()
    else:
        T = np.array(cam_transl)
    T[1] *= -1
    K = np.eye(4)
    K[:3, 3] = T
    return K


def load_pkl_data(pkl_file_path, device='cuda'):
    """Load a pkl file and convert arrays to tensors.

    Args:
        pkl_file_path: Path to the pkl file.
        device: Target device for tensors.

    Returns:
        Dict with tensor values and 'batch_size' key added.
    """
    data = mmcv.load(pkl_file_path)
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            data[key] = torch.as_tensor(np.asarray(data[key]), dtype=torch.float32, device=device)
    data['batch_size'] = data['expression'].shape[0]
    return data


class JointMapper(nn.Module):
    """Map joint indices to a subset."""

    def __init__(self, joint_maps=None):
        super().__init__()
        self.register_buffer(
            'joint_maps', torch.tensor(joint_maps, dtype=torch.long)
        )

    def forward(self, joints, **kwargs):
        return torch.index_select(joints, 1, self.joint_maps)


def get_gtdata(gt_file_path, to6d=False):
    """Load ground truth motion data from a pkl file.

    The pkl file should contain SMPLX parameters: jaw_pose, leye_pose,
    reye_pose, global_orient, body_pose_axis, left_hand_pose,
    right_hand_pose, expression.

    Args:
        gt_file_path: Path to the ground truth pkl file.
        to6d: If True, convert to 6D rotation representation.

    Returns:
        Tensor of shape (1, T, C) on CUDA containing the full pose.
    """
    with open(gt_file_path, 'rb') as f:
        data = pickle.load(f)
    try:
        jaw_pose = as_numpy_float32(data['jaw_pose'])
    except KeyError:
        data = data[0]

    jaw_pose = as_numpy_float32(data['jaw_pose'])
    leye_pose = as_numpy_float32(data['leye_pose'])
    reye_pose = as_numpy_float32(data['reye_pose'])
    global_orient = as_numpy_float32(data['global_orient']).squeeze()
    body_pose = as_numpy_float32(data['body_pose_axis'])
    left_hand_pose = as_numpy_float32(data['left_hand_pose'])
    right_hand_pose = as_numpy_float32(data['right_hand_pose'])
    expression = as_numpy_float32(data['expression'])

    lower = LOWER_POSE.to(torch.float32)
    lower = lower.unsqueeze(0).repeat(data['jaw_pose'].shape[0], 1).numpy()

    # Override eye poses with fixed values
    leye_pose = lower[:, 0:3]
    reye_pose = lower[:, 3:6]

    # Adjust joint3 to account for changed global orientation
    global_rotation_mat = axis_angle_to_matrix(
        torch.as_tensor(np.asarray(global_orient), dtype=torch.float32).reshape(-1, 1, 3)
    )
    node_3_mat = axis_angle_to_matrix(
        torch.as_tensor(np.asarray(body_pose[:, [6, 7, 8]]), dtype=torch.float32).reshape(-1, 1, 3)
    )
    global_mat_node_3 = global_rotation_mat @ node_3_mat

    global_orient = lower[:, 6:9]
    new_global_rotation_mat = axis_angle_to_matrix(
        torch.as_tensor(np.asarray(global_orient), dtype=torch.float32).reshape(-1, 1, 3)
    )
    R_global_inv = new_global_rotation_mat.transpose(-1, -2)
    R_local_3 = torch.matmul(R_global_inv, global_mat_node_3)
    new_local_3 = matrix_to_axis_angle(R_local_3).reshape(-1, 3)

    body_pose[:, [6, 7, 8]] = new_local_3.detach().cpu().numpy()
    body_pose[:, FIXED_BODY_INDICES] = lower[:, 9:]

    full_body = np.concatenate([
        as_numpy_float32(jaw_pose),
        as_numpy_float32(leye_pose),
        as_numpy_float32(reye_pose),
        as_numpy_float32(global_orient),
        as_numpy_float32(body_pose),
        as_numpy_float32(left_hand_pose),
        as_numpy_float32(right_hand_pose),
    ], axis=1)

    hand_dim = right_hand_pose.shape[1]
    if hand_dim == 12:
        full_body = to3d_local(full_body).astype(np.float32)

    if to6d:
        full_body = torch.as_tensor(np.asarray(full_body), dtype=torch.float32)
        full_body = matrix_to_rotation_6d(
            axis_angle_to_matrix(full_body.reshape(-1, 55, 3))
        ).reshape(-1, 330)
        full_body = as_numpy_float32(full_body)

    poses = np.concatenate([as_numpy_float32(full_body), as_numpy_float32(expression)], axis=1)[np.newaxis, ...]
    poses = torch.as_tensor(np.asarray(poses), dtype=torch.float32, device='cuda')
    return poses


def get_ourdata(pred_file_path, to6d=False):
    """Load model prediction data from a pkl file.

    Similar to get_gtdata but handles batched predictions where each
    sample may be stored as a separate batch entry.

    Args:
        pred_file_path: Path to the prediction pkl file.
        to6d: If True, convert to 6D rotation representation.

    Returns:
        Tensor of shape (B, T, C) on CUDA.
    """
    with open(pred_file_path, 'rb') as f:
        data = pickle.load(f)
    try:
        jaw_pose = as_numpy_float32(data['jaw_pose'])
    except (KeyError, TypeError):
        data = data[0]

    jaw_pose = as_numpy_float32(data['jaw_pose'])
    leye_pose = as_numpy_float32(data['leye_pose'])
    reye_pose = as_numpy_float32(data['reye_pose'])
    global_orient = np.array(data['global_orient'])
    body_pose = as_numpy_float32(data['body_pose_axis'])
    left_hand_pose = as_numpy_float32(data['left_hand_pose'])
    right_hand_pose = as_numpy_float32(data['right_hand_pose'])
    expression = as_numpy_float32(data['expression'])

    lower = LOWER_POSE.to(torch.float32)
    lower = lower.unsqueeze(0).unsqueeze(0).repeat(
        data['jaw_pose'].shape[0], data['jaw_pose'].shape[1], 1
    ).numpy()
    body_pose[:, :, FIXED_BODY_INDICES] = lower[:, :, 9:]

    full_body = np.concatenate([
        as_numpy_float32(jaw_pose),
        as_numpy_float32(leye_pose),
        as_numpy_float32(reye_pose),
        as_numpy_float32(global_orient),
        as_numpy_float32(body_pose),
        as_numpy_float32(left_hand_pose),
        as_numpy_float32(right_hand_pose),
    ], axis=2)

    B = full_body.shape[0]
    hand_dim = right_hand_pose.shape[2]

    full_body_list = []
    for i in range(B):
        full_body_tmp = full_body[i]
        if hand_dim == 12:
            full_body_tmp = to3d_local(full_body_tmp).astype(np.float32)
        if to6d:
            full_body_tmp = torch.as_tensor(np.asarray(full_body_tmp), dtype=torch.float32)
            full_body_tmp = matrix_to_rotation_6d(
                axis_angle_to_matrix(full_body_tmp.reshape(-1, 55, 3))
            ).reshape(-1, 330)
            full_body_tmp = as_numpy_float32(full_body_tmp)
        full_body_list.append(full_body_tmp)

    full_body = np.stack(full_body_list, axis=0)
    poses = np.concatenate([as_numpy_float32(full_body), as_numpy_float32(expression)], axis=2)
    poses = torch.as_tensor(np.asarray(poses), dtype=torch.float32, device='cuda')
    return poses


def get_vertices(smplx_model, full_pose, betas):
    """Forward SMPLX model to get mesh vertices.

    Args:
        smplx_model: SMPLX body model instance.
        full_pose: Tensor of shape (B, T, C) containing the full body
            pose parameters (axis-angle + expression).
        betas: Shape parameters tensor.

    Returns:
        List of numpy arrays, each of shape (T, V, 3).
    """
    num = full_pose.shape[0]
    vertices_list = []
    for j in range(num):
        output = smplx_model(
            betas=betas,
            expression=full_pose[j][:, 165:(165 + EXP_DIM)],
            jaw_pose=full_pose[j][:, 0:3],
            leye_pose=full_pose[j][:, 3:6],
            reye_pose=full_pose[j][:, 6:9],
            global_orient=full_pose[j][:, 9:12],
            body_pose=full_pose[j][:, 12:75],
            left_hand_pose=full_pose[j][:, 75:120],
            right_hand_pose=full_pose[j][:, 120:165],
            return_verts=True,
        )
        vertices_list.append(output.vertices.detach().cpu().numpy().squeeze())
    return vertices_list


def render_gt(smplx_model, rendertool, gt_file, audio_path, save_path,
              add_text=False, asr_path=None):
    """Render ground truth motion from a pkl file.

    Args:
        smplx_model: SMPLX body model instance.
        rendertool: RenderTool instance for rendering.
        gt_file: Path to the ground truth pkl file.
        audio_path: Path to the audio file (optional, can be None).
        save_path: Output video file path.
        add_text: Whether to overlay frame number text.
        asr_path: Path to ASR data for text overlay (optional).
    """
    gt_3d = get_gtdata(gt_file, to6d=False)

    with open(gt_file, 'rb') as f:
        betas = pickle.load(f)['betas']
    betas = torch.as_tensor(np.asarray(betas), dtype=torch.float32, device='cuda')

    gt_3d = poses2pred_local(
        gt_3d.squeeze(), stand=False, fix_global=False
    ).unsqueeze(0)

    vertices_list = get_vertices(smplx_model, gt_3d, betas)

    rendertool._render_sequences(
        audio_path, vertices_list, fps=25,
        video_fname=save_path, stand=False, face=False,
        whole_body=False, add_text=add_text, asr_path=asr_path,
    )


def render_ours(smplx_model, rendertool, pred_file, gt_file, audio_path,
                save_path, fix_global=True, add_text=False, asr_path=None):
    """Render model predictions from a pkl file.

    Args:
        smplx_model: SMPLX body model instance.
        rendertool: RenderTool instance for rendering.
        pred_file: Path to the prediction pkl file.
        gt_file: Path to the ground truth pkl file (for betas and alignment).
            If None, betas are read from pred_file and global orientation
            is fixed using the prediction's own mean.
        audio_path: Path to the audio file (optional, can be None).
        save_path: Output video file path.
        fix_global: Whether to fix global orientation to mean value.
        add_text: Whether to overlay frame number text.
        asr_path: Path to ASR data for text overlay (optional).
    """
    pred_3d = get_ourdata(pred_file, to6d=False)

    if gt_file is not None:
        gt_3d = get_gtdata(gt_file, to6d=False)
        # Align sequence length
        L1 = pred_3d.shape[1]
        L2 = gt_3d.shape[1]
        if L1 < L2:
            diff_L = L2 - L1
            pred_3d = torch.cat([pred_3d, pred_3d[:, -diff_L:, :]], dim=1)
        elif L1 > L2:
            pred_3d = pred_3d[:, :L2, :]

        if fix_global:
            mean_global = torch.mean(
                gt_3d[:, :, 9:12], dim=1, keepdim=True
            ).repeat(1, pred_3d.shape[1], 1)
            pred_3d[:, :, 9:12] = mean_global

        with open(gt_file, 'rb') as f:
            betas = pickle.load(f)['betas']
    else:
        # No GT file: fix global orientation to default front-facing pose
        if fix_global:
            fixed = FIXED_GLOBAL_ORIENT.to(pred_3d.device).reshape(1, 1, 3)
            pred_3d[:, :, 9:12] = fixed.expand_as(pred_3d[:, :, 9:12])

        with open(pred_file, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, list):
                data = data[0]
            betas = data['betas']

    betas = torch.as_tensor(np.asarray(betas), dtype=torch.float32, device='cuda')

    vertices_list = get_vertices(smplx_model, pred_3d, betas)

    rendertool._render_sequences(
        audio_path, vertices_list, fps=25,
        video_fname=save_path, stand=False, face=False,
        whole_body=False, add_text=add_text, asr_path=asr_path,
    )


def build_smplx_model(smplx_model_path=None):
    """Build and return an SMPLX model on CUDA.

    Args:
        smplx_model_path: Path to SMPLX_NEUTRAL.npz. If None, uses the
            default path under render_model/smplx/.

    Returns:
        SMPLX model on CUDA.
    """
    patch_smplx_create_mean_pose()
    if smplx_model_path is None:
        smplx_model_path = os.path.join(
            _SCRIPT_DIR, 'render_model', 'smplx', 'SMPLX_NEUTRAL.npz'
        )
    model_params = dict(
        model_path=smplx_model_path,
        model_type='smplx',
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        num_betas=10,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        use_pca=False,
        flat_hand_mean=True,
        create_expression=True,
        num_expression_coeffs=EXP_DIM,
        num_pca_comps=12,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=False,
        dtype=torch.float32,
    )
    return smplx.create(**model_params).to('cuda')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render SMPLX body motion from pkl files.'
    )
    parser.add_argument(
        '--pkl_file', type=str, required=True,
        help='Path to the input pkl file (ground truth or prediction).',
    )
    parser.add_argument(
        '--gt_file', type=str, default=None,
        help='Path to the ground truth pkl file. Required for mode=ours.',
    )
    parser.add_argument(
        '--save_path', type=str, default='output.mp4',
        help='Output video file path.',
    )
    parser.add_argument(
        '--audio', type=str, default=None,
        help='Path to the audio file to overlay on the video.',
    )
    parser.add_argument(
        '--mode', type=str, default='gt', choices=['gt', 'ours'],
        help='Rendering mode: "gt" for ground truth, "ours" for predictions.',
    )
    parser.add_argument(
        '--smplx_model_path', type=str, default=None,
        help='Path to SMPLX_NEUTRAL.npz model file.',
    )
    parser.add_argument(
        '--add_text', action='store_true',
        help='Overlay frame number text on the rendered video.',
    )
    parser.add_argument(
        '--asr_path', type=str, default=None,
        help='Path to ASR data file for text overlay.',
    )
    parser.add_argument(
        '--fix_global', action='store_true', default=True,
        help='Fix global orientation to mean GT value (for mode=ours).',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("Initializing SMPLX model...")
    smplx_model = build_smplx_model(args.smplx_model_path)

    print("Initializing renderer...")
    rendertool = RenderTool()

    if args.mode == 'gt':
        print(f"Rendering GT motion from: {args.pkl_file}")
        render_gt(
            smplx_model, rendertool,
            gt_file=args.pkl_file,
            audio_path=args.audio,
            save_path=args.save_path,
            add_text=args.add_text,
            asr_path=args.asr_path,
        )
    elif args.mode == 'ours':
        print(f"Rendering predictions from: {args.pkl_file}")
        gt_file = args.gt_file
        if gt_file is None:
            with open(args.pkl_file, 'rb') as f:
                _pred_data = pickle.load(f)
                if isinstance(_pred_data, list):
                    _pred_data = _pred_data[0]
                gt_file = _pred_data.get('gt_path', None)
            if gt_file is not None:
                print(f"Using gt_path from pkl: {gt_file}")
        render_ours(
            smplx_model, rendertool,
            pred_file=args.pkl_file,
            gt_file=gt_file,
            audio_path=args.audio,
            save_path=args.save_path,
            fix_global=args.fix_global,
            add_text=args.add_text,
            asr_path=args.asr_path,
        )

    print(f"Video saved to: {args.save_path}")
