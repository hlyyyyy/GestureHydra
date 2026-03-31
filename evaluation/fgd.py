import os
import sys
import pickle
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
import torch
import numpy as np

from nets.init_model import init_model
from evaluation.FGD import EmbeddingSpaceEvaluator
from data_utils.rotation_conversion import axis_angle_to_matrix, matrix_to_rotation_6d

# Script directory for resolving relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Compute controllable joint indices (exclude fixed lower body joints)
fix_index_3d = [0,1,2,3,4,5,6,7,8,12,13,14,15,16,17,
                21,22,23,24,25,26,
                38,31,32,33,34,35,
                39,40,41,42,43,44]  # 11x3 without global orient
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
        c_index_6d.append(2*i)
        c_index_6d.append(2 * i + 1)
    i = i + 1
c_index_6d = np.asarray(c_index_6d)

c_index = c_index_6d

# Load hand PCA components
with open(os.path.join(SCRIPT_DIR, 'config/hand_component.json')) as file_obj:
    comp = json.load(file_obj)
    left_hand_c = np.asarray(comp['left'])
    right_hand_c = np.asarray(comp['right'])

def to3d_local(data):
    left_hand_pose = np.einsum('bi,ij->bj', data[:, 75:87], left_hand_c[:12, :])
    right_hand_pose = np.einsum('bi,ij->bj', data[:, 87:99], right_hand_c[:12, :])
    data = np.concatenate((data[:, :75], left_hand_pose, right_hand_pose), axis=-1)
    return data

def find_pkl_files(directory):
    """Find all .pkl files in a directory and its subdirectories."""
    pkl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    return pkl_files


def get_gtdata(gt_file_path):

    with open(gt_file_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, list):
        data = data[0]

    betas = np.array(data['betas'])  # (1, 300)
    jaw_pose = np.array(data['jaw_pose'])    # (253, 3)
    leye_pose = np.array(data['leye_pose'])  # (253, 3)
    reye_pose = np.array(data['reye_pose'])  # (253, 3)
    global_orient = np.array(data['global_orient']).squeeze()  # (253, 3)
    body_pose = np.array(data['body_pose_axis'])  # (253, 63)
    left_hand_pose = np.array(data['left_hand_pose'])    # (253, 12)
    right_hand_pose = np.array(data['right_hand_pose'])  # (253, 12)

    expression = np.array(data['expression'])
    full_body = np.concatenate((jaw_pose, leye_pose, reye_pose, global_orient, body_pose, left_hand_pose, right_hand_pose), axis=1)  # (253, 99)

    hand_dim = right_hand_pose.shape[1]
    if hand_dim == 12:
        full_body = to3d_local(full_body).astype(np.float32)
    full_body = torch.from_numpy(full_body)  # torch.Size([253, 165])
    full_body = matrix_to_rotation_6d(axis_angle_to_matrix(full_body.reshape(-1, 55, 3))).reshape(-1, 330)  # torch.Size([253, 330])
    full_body = np.asarray(full_body)

    return full_body, expression

def get_preddata(pred_file_path):

    with open(pred_file_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, list):
        data = data[0]

    betas = np.array(data['betas'])  # (1, 300)
    jaw_pose = np.array(data['jaw_pose'])    # (253, 3)
    leye_pose = np.array(data['leye_pose'])  # (253, 3)
    reye_pose = np.array(data['reye_pose'])  # (253, 3)
    global_orient = np.array(data['global_orient']).squeeze()  # (253, 3)
    body_pose = np.array(data['body_pose_axis'])  # (253, 63)
    left_hand_pose = np.array(data['left_hand_pose'])    # (253, 12)
    right_hand_pose = np.array(data['right_hand_pose'])  # (253, 12)

    expression = np.array(data['expression'])
    full_body = np.concatenate((jaw_pose, leye_pose, reye_pose, global_orient, body_pose, left_hand_pose, right_hand_pose), axis=2)  # (253, 99)

    B = full_body.shape[0]
    hand_dim = right_hand_pose.shape[2]

    full_body_list = []
    for i in range(B):
        full_body_tmp = full_body[i]

        if hand_dim == 12:
            full_body = to3d_local(full_body_tmp).astype(np.float32)
        full_body_tmp = torch.from_numpy(full_body_tmp)  # torch.Size([253, 165])
        full_body_tmp = matrix_to_rotation_6d(axis_angle_to_matrix(full_body_tmp.reshape(-1, 55, 3))).reshape(-1, 330)  # torch.Size([253, 330])
        full_body_tmp = np.asarray(full_body_tmp)

        full_body_list.append(full_body_tmp)

    full_body = np.stack(full_body_list, axis=0)
    return full_body, expression


def test(FGD_handler, pkl_root, gt_root, fgd_model_body_path):
    print('start testing')

    norm_stats_fn = os.path.join(SCRIPT_DIR, 'data_utils/norm_stats.npy')
    norm_stats = np.load(norm_stats_fn, allow_pickle=True)
    norm_stats_torch = [
        torch.from_numpy(norm_stats[0]).to('cuda'),
        torch.from_numpy(norm_stats[1]).to('cuda'),
    ]
    norm_stats = norm_stats_torch

    c_index = c_index_6d

    pkl_list = find_pkl_files(pkl_root)

    print("total num of files: ", len(pkl_list))

    for i in tqdm(range(len(pkl_list))):
        pred_file_path = pkl_list[i]
        gt_file_path = pred_file_path.replace(pkl_root, gt_root).replace(".pkl", ".pkl")

        poses, exp = get_gtdata(gt_file_path)
        poses = torch.from_numpy(poses[np.newaxis, :].transpose((0, 2, 1))).to('cuda')
        exp = torch.from_numpy(exp[np.newaxis, :].transpose((0, 2, 1))).to('cuda')

        jaw = poses[:, :6]
        gt_poses = torch.cat([jaw, poses[:, c_index], exp], dim=1)  # 376
        poses = torch.cat([poses, exp], dim=1)  # 430
        gt_poses = poses[:, c_index]

        pred_poses, pred_exp = get_preddata(pred_file_path)
        pred_poses = torch.from_numpy(pred_poses.transpose((0, 2, 1))).to('cuda')
        pred_exp = torch.from_numpy(pred_exp.transpose((0, 2, 1))).to('cuda')

        pred_jaw = pred_poses[:, :6]
        pred = torch.cat([pred_jaw, pred_poses[:, c_index], pred_exp], dim=1)  # 376
        pred = pred_poses[:, c_index]

        # pred: (B, 264, T), gt_poses: (1, 264, T) - using unnormalized data
        FGD_handler.push_samples(pred[:].transpose(1, 2).unfold(1, 90, 90).flatten(0, 1),
                                 gt_poses[0:1].transpose(1, 2).unfold(1, 90, 90).flatten(0, 1))

    print()
    fgd_dist, feat_dist = FGD_handler.get_scores('bh')

    print('body_fgd=', fgd_dist.item())
    print('feat_dist=', feat_dist.item())
    print("test on=", pkl_root)
    print("test model=", fgd_model_body_path)


from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--pkl_path', required=True, type=str,
                        help='Path to predicted gesture pkl files')
    parser.add_argument('--gt_path', required=True, type=str,
                        help='Path to ground truth gesture pkl files')
    parser.add_argument('--fgd_model', type=str,
                        default=os.path.join(SCRIPT_DIR, 'checkpoints/fgd.pth'),
                        help='Path to FGD model checkpoint')
    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.cuda.set_device(device)

    fgd_model_body_path = args.fgd_model

    print('init fgd evaluator...')
    body_ae = init_model(args, True, fgd_model_body_path)
    FGD_handler = EmbeddingSpaceEvaluator(body_ae, None, 'cuda')

    pkl_root = args.pkl_path
    gt_root = args.gt_path
    test(FGD_handler, pkl_root, gt_root, fgd_model_body_path)


if __name__ == '__main__':
    main()
