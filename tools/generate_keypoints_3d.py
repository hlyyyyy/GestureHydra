"""Extract SMPL-X 3D keypoints from gesture pkl files in the streamer-dataset.

For each pkl file under  data/streamer-dataset/{split}/gestures/,
run SMPL-X forward kinematics and save the resulting 3D joint positions as:
    data/streamer-dataset/{split}/keypoints_3d/{anchor_id}/{video_md5}/{start}_{end}.npy

Each .npy file has shape (T, 144, 3)  -- 144 SMPL-X joints (body + hands + face).

Usage:
    python tools/generate_keypoints_3d.py \
        --dataset_root data/streamer-dataset \
        --body_model_path body_models \
        --splits train test_seen test_unseen \
        --batch_size 8 \
        --device cuda

Prerequisites:
    pip install smplx torch
    The SMPL-X body model files must be placed under --body_model_path/smplx/.
"""

import argparse
import copy
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# SMPL-X model setup (adapted from gesturehydra/models/utils/smplx2joints.py)
# We inline the minimal logic here to avoid heavy imports from the main
# codebase (mmcv / mmgen / etc.) that are unnecessary for preprocessing.
# ---------------------------------------------------------------------------

def build_smplx_layer(body_model_path: str, device: torch.device):
    """Build and return a frozen SMPL-X body model layer."""
    import smplx as smplx_lib

    model_path = os.path.join(
        body_model_path, "smplx", "SMPLX_MALE_shape2019_exp2020.npz"
    )
    layer = smplx_lib.create(
        model_path=model_path,
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
        flat_hand_mean=True,
        use_face_contour=True,
        use_pca=False,
        use_hands=True,
        use_face=True,
        num_betas=10,
        num_pca_comps=12,
        num_expression_coeffs=100,
        dtype=torch.float32,
    ).to(device)
    layer.eval()
    for p in layer.parameters():
        p.requires_grad_(False)
    return layer


# SMPL-X with use_face_contour=True outputs 144 joints
NUM_JOINTS = 144


# ---------------------------------------------------------------------------
# PKL loading -- convert streamer-dataset pkl to SMPL-X input tensors
# ---------------------------------------------------------------------------

def load_pkl(pkl_path: str):
    """Load a gesture pkl and return a dict of numpy arrays."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        data = data[0]
    return data


def pkl_to_smplx_params(data: dict):
    """Extract SMPL-X parameters from a single pkl dict.

    Returns a dict of numpy arrays with the naming convention used by
    the smplx body model layer.
    """
    jaw_pose = np.array(data["jaw_pose"])           # (T, 3)
    leye_pose = np.array(data["leye_pose"])         # (T, 3)
    reye_pose = np.array(data["reye_pose"])         # (T, 3)
    global_orient = np.array(data["global_orient"]).squeeze()  # (T, 3)
    body_pose = np.array(data["body_pose_axis"])    # (T, 63)
    left_hand_pose = np.array(data["left_hand_pose"])   # (T, 45)
    right_hand_pose = np.array(data["right_hand_pose"]) # (T, 45)
    betas = np.array(data["betas"])                 # (1, 10) or (10,)
    transl = np.array(data["transl"])               # (T, 3)

    # expression may or may not be present
    if "expression" in data:
        expression = np.array(data["expression"])   # (T, 50)
    else:
        expression = np.zeros((jaw_pose.shape[0], 50), dtype=np.float32)

    if betas.ndim == 1:
        betas = betas[np.newaxis, :]  # (1, 10)

    return {
        "jaw_pose": jaw_pose,
        "leye_pose": leye_pose,
        "reye_pose": reye_pose,
        "global_orient": global_orient,
        "body_pose": body_pose,
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
        "betas": betas,
        "transl": transl,
        "expression": expression,
    }


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_joints(smplx_layer, params: dict, device: torch.device):
    """Run SMPL-X forward kinematics and return 3D joints.

    Args:
        smplx_layer: the SMPL-X body model on `device`.
        params: dict of numpy arrays from `pkl_to_smplx_params`.
        device: torch device.

    Returns:
        joints: numpy array of shape (T, 144, 3).
    """
    T = params["jaw_pose"].shape[0]
    betas_np = params["betas"]  # (1, 10)

    def to_t(arr):
        return torch.from_numpy(arr).float().to(device)

    output = smplx_layer(
        return_verts=False,
        betas=to_t(np.broadcast_to(betas_np, (T, 10))),
        jaw_pose=to_t(params["jaw_pose"]),
        leye_pose=to_t(params["leye_pose"]),
        reye_pose=to_t(params["reye_pose"]),
        global_orient=to_t(params["global_orient"]),
        body_pose=to_t(params["body_pose"]),
        left_hand_pose=to_t(params["left_hand_pose"]),
        right_hand_pose=to_t(params["right_hand_pose"]),
        expression=to_t(params["expression"]),
        transl=to_t(params["transl"]),
    )

    # output.joints: (T, >=144, 3) -- take first 144
    joints = output.joints[:, :NUM_JOINTS, :].cpu().numpy()  # (T, 144, 3)

    # Add translation (already applied by transl param, but the original
    # pipeline adds it explicitly -- check if already included)
    # NOTE: smplx.create with create_transl=True already applies transl
    # inside the forward pass, so we do NOT add it again here.

    return joints


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract SMPL-X 3D keypoints from streamer-dataset."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/streamer-dataset",
    )
    parser.add_argument(
        "--body_model_path",
        type=str,
        default="body_models",
        help="Directory containing smplx/ model files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test_seen", "test_unseen"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    device = torch.device(args.device)

    print(f"Loading SMPL-X body model from {args.body_model_path} ...")
    smplx_layer = build_smplx_layer(args.body_model_path, device)

    for split in args.splits:
        gesture_dir = dataset_root / split / "gestures"
        kp3d_dir = dataset_root / split / "keypoints_3d"

        if not gesture_dir.exists():
            print(f"[SKIP] {gesture_dir} not found")
            continue

        pkl_files = sorted(gesture_dir.rglob("*.pkl"))
        print(f"\n[{split}] Found {len(pkl_files)} pkl files")

        for pkl_path in tqdm(pkl_files, desc=split):
            # gestures/{anchor}/{md5}/{clip}.pkl -> keypoints_3d/{anchor}/{md5}/{clip}.npy
            rel = pkl_path.relative_to(gesture_dir)
            save_path = kp3d_dir / rel.with_suffix(".npy")

            if save_path.exists():
                continue

            try:
                data = load_pkl(str(pkl_path))
                params = pkl_to_smplx_params(data)
                joints = extract_joints(smplx_layer, params, device)

                os.makedirs(save_path.parent, exist_ok=True)
                np.save(str(save_path), joints)
            except Exception as e:
                print(f"[ERROR] {pkl_path}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
