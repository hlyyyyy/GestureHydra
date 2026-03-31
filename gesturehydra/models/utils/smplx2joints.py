import sys
import os

import numpy as np
import copy
import smplx
import os.path as osp
import pickle
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ['PYOPENGL_PLATFORM'] = 'egl'
comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


HUMAN_MODEL_PATH = 'body_models'
DEFAULT_SMPLX_CONFIG = dict(
    create_global_orient=True,
    create_body_pose=True,
    create_betas=True,
    create_left_hand_pose=True,
    create_right_hand_pose=True,
    create_expression=True,
    create_jaw_pose=True,
    create_leye_pose=True,
    create_reye_pose=True,
    create_transl=True,
)

def smpl_to_openpose(
    use_face=True,
    use_hands=True,
    use_face_contour=True,
) -> np.ndarray:
    # smplx-->openpose_body_25
    body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                            8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                            63, 64, 65], dtype=np.int32)  # 25
    mapping = [body_mapping]

    if use_hands:
        lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                  67, 28, 29, 30, 68, 34, 35, 36, 69,
                                  31, 32, 33, 70], dtype=np.int32)
        rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                  43, 44, 45, 73, 49, 50, 51, 74, 46,
                                  47, 48, 75], dtype=np.int32)
        mapping += [lhand_mapping, rhand_mapping]

    if use_face:
        face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                 dtype=np.int32)
        mapping += [face_mapping]
    return np.concatenate(mapping)


class JointMapper(nn.Module):

    def __init__(self, joint_maps=None):
        super().__init__()
        self.register_buffer('joint_maps', torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        return torch.index_select(joints, 1, self.joint_maps)


def load_smplx_model(device='cpu', **kwargs):
    body_model = smplx.create(
        **DEFAULT_SMPLX_CONFIG,
        **kwargs).to(device=device)
    return body_model


class SMPLX(object):
    def __init__(self):
        self.layer_arg = {'model_path': f'{HUMAN_MODEL_PATH}/smplx/SMPLX_MALE_shape2019_exp2020.npz', 
                          'flat_hand_mean': True, 'use_face_contour': True, 'use_pca': False, 
                          'use_hands': True, 'use_face': True, 'num_betas': 10, 'num_pca_comps': 12, 'num_expression_coeffs': 100}
        self.layer = {
            'neutral': load_smplx_model(dtype=torch.float32, **self.layer_arg)
        }
        self.vertex_num = 10475
        self.face = self.layer['neutral'].faces
        self.shape_param_dim = 10
        self.expr_code_dim = 10
        with open(osp.join(HUMAN_MODEL_PATH, 'smplx', 'SMPLX_to_J14.pkl'), 'rb') as f:
            self.j14_regressor = pickle.load(f, encoding='latin1')
        with open(osp.join(HUMAN_MODEL_PATH, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            self.hand_vertex_idx = pickle.load(f, encoding='latin1')
        self.face_vertex_idx = np.load(
            osp.join(HUMAN_MODEL_PATH, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))
        self.J_regressor = self.layer['neutral'].J_regressor.numpy()
        self.J_regressor_idx = {'pelvis': 0,
                                'lwrist': 20, 'rwrist': 21, 'neck': 12}
        self.orig_hand_regressor = self.make_hand_regressor()

        # original SMPLX joint set
        # 22 (body joints) + 30 (hand joints) + 1 (face jaw joint)
        self.orig_joint_num = 53
        self.orig_joints_name = \
            ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',  # body joints
             'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',  # left hand joints
             'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',  # right hand joints
             'Jaw'  # face jaw joint
             )
        self.orig_flip_pairs = \
            ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),  # body joints
             (22, 37), (23, 38), (24, 39), (25, 40), (26, 41), (27, 42), (28, 43), (29, 44), (30,
                                                                                              45), (31, 46), (32, 47), (33, 48), (34, 49), (35, 50), (36, 51)  # hand joints
             )
        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')
        self.orig_joint_part = \
            {'body': range(self.orig_joints_name.index('Pelvis'), self.orig_joints_name.index('R_Wrist')+1),
             'lhand': range(self.orig_joints_name.index('L_Index_1'), self.orig_joints_name.index('L_Thumb_3')+1),
             'rhand': range(self.orig_joints_name.index('R_Index_1'), self.orig_joints_name.index('R_Thumb_3')+1),
             'face': range(self.orig_joints_name.index('Jaw'), self.orig_joints_name.index('Jaw')+1)}

        # changed SMPLX joint set for the supervision
        # 25 (body joints) + 40 (hand joints) + 72 (face keypoints)
        self.joint_num = 137
        self.joints_name = \
            ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',  # body joints
             'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',  # left hand joints
             'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',  # right hand joints
             # face keypoints (too many keypoints... omit real names. have same name of keypoints defined in FLAME class)
             *['Face_' + str(i) for i in range(1, 73)]
             )
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.lwrist_idx = self.joints_name.index('L_Wrist')
        self.rwrist_idx = self.joints_name.index('R_Wrist')
        self.neck_idx = self.joints_name.index('Neck')
        self.flip_pairs = \
            ((1, 2), (3, 4), (5, 6), (8, 9), (10, 11), (12, 13), (14, 17), (15, 18), (16, 19), (20, 21), (22, 23),  # body joints
             (25, 45), (26, 46), (27, 47), (28, 48), (29, 49), (30, 50), (31, 51), (32, 52), (33, 53), (34, 54), (35,
                                                                                                                  55), (36, 56), (37, 57), (38, 58), (39, 59), (40, 60), (41, 61), (42, 62), (43, 63), (44, 64),  # hand joints
                (67, 68),  # face eyeballs
                (69, 78), (70, 77), (71, 76), (72, 75), (73, 74),  # face eyebrow
                (83, 87), (84, 86),  # face below nose
                (88, 97), (89, 96), (90, 95), (91,
                                               94), (92, 99), (93, 98),  # face eyes
                (100, 106), (101, 105), (102, 104), (107,
                                                     111), (108, 110),  # face mouth
                (112, 116), (113, 115), (117, 119),  # face lip
                (120, 136), (121, 135), (122, 134), (123, 133), (124,
                                                                 132), (125, 131), (126, 130), (127, 129)  # face contours
             )
        # self.joint_idx = \
        # (0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55, # body joints
        # 37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70, # left hand joints
        # 52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75, # right hand joints
        # 22,15, # jaw, head #2
        # 57,56, # eyeballs #2
        # 76,77,78,79,80,81,82,83,84,85, # eyebrow #10
        # 86,87,88,89, # nose #4
        # 90,91,92,93,94, # below nose # 5
        # 95,96,97,98,99,100,101,102,103,104,105,106, # eyes # 12
        # 107, # right mouth # 1
        # 108,109,110,111,112, # upper mouth # 5
        # 113, # left mouth # 1
        # 114,115,116,117,118, # lower mouth # 5
        # 119, # right lip # 1
        # 120,121,122, # upper lip # 3
        # 123, # left lip # 1
        # 124,125,126, # lower lip # 3
        # 127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143 # face contour # 17
        # )
        self.joint_idx = range(144)
        self.joint_part = \
            {'body': range(self.joints_name.index('Pelvis'), self.joints_name.index('Nose')+1),
             'lhand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('L_Pinky_4')+1),
             'rhand': range(self.joints_name.index('R_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),
             'hand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),
             'face': range(self.joints_name.index('Face_1'), self.joints_name.index('Face_72')+1)}

        # changed SMPLX joint set for PositionNet prediction
        self.pos_joint_num = 65  # 25 (body joints) + 40 (hand joints)
        self.pos_joints_name = \
            ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',  # body joints
             'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',  # left hand joints
             'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',  # right hand joints
             )
        self.pos_joint_part = \
            {'body': range(self.pos_joints_name.index('Pelvis'), self.pos_joints_name.index('Nose')+1),
             'lhand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('L_Pinky_4')+1),
             'rhand': range(self.pos_joints_name.index('R_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1),
             'hand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1)}
        self.pos_joint_part['L_MCP'] = [self.pos_joints_name.index('L_Index_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index(
                                            'L_Middle_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index(
                                            'L_Ring_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Pinky_1') - len(self.pos_joint_part['body'])]
        self.pos_joint_part['R_MCP'] = [self.pos_joints_name.index('R_Index_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index(
                                            'R_Middle_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index(
                                            'R_Ring_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Pinky_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand'])]

    def make_hand_regressor(self):
        regressor = self.layer['neutral'].J_regressor.numpy()
        lhand_regressor = np.concatenate((regressor[[20, 37, 38, 39], :],
                                          np.eye(self.vertex_num)[5361, None],
                                          regressor[[25, 26, 27], :],
                                          np.eye(self.vertex_num)[4933, None],
                                          regressor[[28, 29, 30], :],
                                          np.eye(self.vertex_num)[5058, None],
                                          regressor[[34, 35, 36], :],
                                          np.eye(self.vertex_num)[5169, None],
                                          regressor[[31, 32, 33], :],
                                          np.eye(self.vertex_num)[5286, None]))
        rhand_regressor = np.concatenate((regressor[[21, 52, 53, 54], :],
                                          np.eye(self.vertex_num)[8079, None],
                                          regressor[[40, 41, 42], :],
                                          np.eye(self.vertex_num)[7669, None],
                                          regressor[[43, 44, 45], :],
                                          np.eye(self.vertex_num)[7794, None],
                                          regressor[[49, 50, 51], :],
                                          np.eye(self.vertex_num)[7905, None],
                                          regressor[[46, 47, 48], :],
                                          np.eye(self.vertex_num)[8022, None]))
        hand_regressor = {'left': lhand_regressor, 'right': rhand_regressor}
        return hand_regressor

    def reduce_joint_set(self, joint):
        new_joint = []
        for name in self.pos_joints_name:
            idx = self.joints_name.index(name)
            new_joint.append(joint[:, idx, :])
        new_joint = torch.stack(new_joint, 1)
        return new_joint


def get_smplx_layer(device):
    smplx_model = SMPLX()
    smplx_layer = copy.deepcopy(smplx_model.layer['neutral']).to(device)
    return smplx_layer, smplx_model


def get_train_joint_idx():
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
    return c_index_3d, c_index_6d

def process_smplx_custom_data(
    smplx_data, 
    smplx_layer, 
    smplx_model, 
    device,
    face_carnical=False, 
    pose=None
):
    '''
    INPUT:
    smplx_data: (bs, frames, 322)

    RETURN:
    vertices: 
    joints: 
    pose: 
    faces:  

    '''
    pose = smplx_data
    batch_size = pose['pose_jaw'].shape[0]
    num_frames = pose['pose_jaw'].shape[1]

    zero_pose = torch.zeros((batch_size, num_frames, 3)
                            ).float().to(device)  # eye poses
    zero_expr = torch.zeros((batch_size, num_frames, 10)).float().to(
        device)  # smplx face expression

    # concat poses
    zero_pose = zero_pose.reshape(batch_size*num_frames, -1)
    zero_expr = zero_expr.reshape(batch_size*num_frames, -1)
    
    pose['betas'] = pose['betas'].repeat(1, num_frames, 1)

    body_parms = {
        # controls the global root orientation 3
        'root_orient': pose['root_orient'].reshape(batch_size*num_frames, -1).to(device),
        # controls the body 63
        'pose_body': pose['pose_body'].reshape(batch_size*num_frames, -1).to(device),
        # controls the finger articulation 90
        'pose_hand': pose['pose_hand'].reshape(batch_size*num_frames, -1).to(device),
        # controls the yaw pose 3
        'pose_jaw': pose['pose_jaw'].reshape(batch_size*num_frames, -1).to(device),
        # controls the global body position 3
        'trans': pose['trans'].reshape(batch_size*num_frames, -1).to(device),
        # controls the body shape. Body shape is static 10
        'betas': pose['betas'].reshape(batch_size*num_frames, -1).to(device),
        'leye': pose['leye'].reshape(batch_size*num_frames, -1).to(device),
        'reye': pose['leye'].reshape(batch_size*num_frames, -1).to(device),
        'expression': pose['expression'].reshape(batch_size*num_frames, -1).to(device),
    }

    output = smplx_layer(
        betas=body_parms['betas'], 
        body_pose=body_parms['pose_body'], 
        global_orient=body_parms['root_orient'],
        left_hand_pose=body_parms['pose_hand'][..., :45], 
        right_hand_pose=body_parms['pose_hand'][..., 45:],
        jaw_pose=body_parms['pose_jaw'], 
        leye_pose=body_parms['leye'], # zero_pose, 
        reye_pose=body_parms['reye'], # zero_pose, 
        expression=body_parms['expression'], # zero_expr,
        transl=body_parms['trans'])
    del zero_pose
    del zero_expr

    vertices = output.vertices.reshape(batch_size, num_frames, 10475, 3)
    joints = output.joints[:, smplx_model.joint_idx, :].reshape(
        batch_size, num_frames, len(smplx_model.joint_idx), 3)
    faces = smplx_model.face
    return vertices, joints, pose, faces

