import torch
import copy
import numpy as np
from ..builder import ARCHITECTURES, build_loss, build_submodule
from ..utils.gaussian_diffusion import (
    GaussianDiffusion, LossType, ModelMeanType, ModelVarType, SpacedDiffusion,
    create_named_schedule_sampler, get_named_beta_schedule, space_timesteps)
from .base_architecture import BaseArchitecture
from mmcv.runner import auto_fp16
from gesturehydra.models.utils.smplx2joints import SMPLX, get_train_joint_idx
from gesturehydra.datasets.audio_smplx_dataset.rotation_conversion import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle)



def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def build_diffusion(cfg):
    beta_scheduler = cfg['beta_scheduler']
    diffusion_steps = cfg['diffusion_steps']

    betas = get_named_beta_schedule(beta_scheduler, diffusion_steps)
    model_mean_type = {
        'start_x': ModelMeanType.START_X,
        'previous_x': ModelMeanType.PREVIOUS_X,
        'epsilon': ModelMeanType.EPSILON
    }[cfg['model_mean_type']]
    model_var_type = {
        'learned': ModelVarType.LEARNED,
        'fixed_small': ModelVarType.FIXED_SMALL,
        'fixed_large': ModelVarType.FIXED_LARGE,
        'learned_range': ModelVarType.LEARNED_RANGE
    }[cfg['model_var_type']]
    if cfg.get('respace', None) is not None:
        diffusion = SpacedDiffusion(use_timesteps=space_timesteps(
            diffusion_steps, cfg['respace']),
                                    betas=betas,
                                    model_mean_type=model_mean_type,
                                    model_var_type=model_var_type,
                                    loss_type=LossType.MSE)
    else:
        diffusion = GaussianDiffusion(betas=betas,
                                      model_mean_type=model_mean_type,
                                      model_var_type=model_var_type,
                                      loss_type=LossType.MSE)
    return diffusion




def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


@ARCHITECTURES.register_module()
class MotionDiffusionA2G(BaseArchitecture):

    def __init__(self,
                 model=None,
                 loss_recon=None,
                 loss_reduction="frame",
                 diffusion_train=None,
                 diffusion_test=None,
                 sampler_type='uniform',
                 init_cfg=None,
                 inference_type='ddpm',
                 initialize_smplx_model=False,
                 loss_kp_3d=None,
                 use_vel_loss=False,
                 loss_vel=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.model = build_submodule(model)
        self.loss_recon = build_loss(loss_recon)
        self.diffusion_train = build_diffusion(diffusion_train)
        self.diffusion_test = build_diffusion(diffusion_test)
        self.sampler = create_named_schedule_sampler(sampler_type,
                                                     self.diffusion_train)
        self.inference_type = inference_type
        self.loss_reduction = loss_reduction
        self.initialize_smplx_model = initialize_smplx_model
        self.use_vel_loss = use_vel_loss
        self.fp16_enabled = False

        if self.initialize_smplx_model:
            self.smplx_model = SMPLX()
            self.smplx_layer = copy.deepcopy(self.smplx_model.layer['neutral'])
            set_requires_grad(self.smplx_layer, False)
            self.kp_3d_criterion = build_loss(loss_kp_3d)
        if self.use_vel_loss:
            self.vel_criterion = build_loss(loss_vel)


    @auto_fp16()
    def forward(self, **kwargs):
        motion = kwargs['motion'].float()
        motion_mask = kwargs['motion_mask'].float()
        motion_length = kwargs['motion_length']
        
        if 'seed' in kwargs.keys():
            seed = kwargs['seed'].float()
            prev_seed = None
            last_seed = None
        elif 'prev_seed' in kwargs.keys():
            seed = None
            prev_seed = kwargs['prev_seed'].float()
            last_seed = kwargs['last_seed'].float()

        if 'wavlm' in kwargs.keys():
            wavlm = kwargs['wavlm'].float()
        
        num_intervals = kwargs.get('num_intervals', 1)
        B, T = motion.shape[:2]

        if self.training:
            t, _ = self.sampler.sample(B, motion.device)
            model_kwargs={
                'motion_mask': motion_mask,
                'motion_length': motion_length,
                'seed': seed,
                'prev_seed': prev_seed,
                'last_seed': last_seed,
                'wavlm': wavlm,
                'num_intervals': num_intervals,
                'motion_style': kwargs['motion_style'] if 'motion_style' in kwargs.keys() else None,
                'motion_style_mask': kwargs['motion_style_mask'] if 'motion_style_mask' in kwargs.keys() else None,
                'key_frame': kwargs['key_frame'] if 'key_frame' in kwargs.keys() else None,
                'betas': kwargs['betas'] if 'betas' in kwargs.keys() else None,
            }
            output = self.diffusion_train.training_losses(model=self.model,
                                                          x_start=motion,
                                                          t=t,
                                                          model_kwargs=model_kwargs)
            pred, target = output['pred'], output['target']
            recon_loss = self.loss_recon(pred, target, reduction_override='none')
            if self.use_vel_loss:
                vel_pred = pred[:, 1:] - pred[:, :-1]
                vel_gt = target[:, 1:] - target[:, :-1]
                vel_loss = self.vel_criterion(vel_pred, vel_gt, reduction_override='none')
            
            if kwargs.get('score', None) is not None:
                # Down-weight loss for low-confidence (occluded) regions
                bs, ts, _ = recon_loss.shape
                recon_loss = recon_loss.reshape(bs, ts, -1, 2)
                score_weights = (kwargs['score'] > 3.).to(recon_loss.dtype)
                recon_loss *= score_weights.unsqueeze(-1)
                recon_loss = recon_loss.reshape(bs, ts, -1)

            # Down-weight loss on keyframe positions
            key_weights = torch.ones([recon_loss.shape[0], recon_loss.shape[1]]).to(recon_loss.device)
            if 'key_frame' in kwargs.keys():
                key_frame = kwargs['key_frame']
                key_weights[key_frame.to(torch.bool)] *= 0.1

            recon_loss = recon_loss.mean(dim=-1) * motion_mask
            recon_loss = recon_loss * key_weights

            recon_loss_batch = \
                recon_loss.sum(dim=1) / motion_mask.sum(dim=1)
            recon_loss_frame = \
                recon_loss.sum() / (motion_mask * key_weights).sum()
            if self.loss_reduction == "frame":
                recon_loss = recon_loss_frame
            else:
                recon_loss = recon_loss_batch
            recon_loss = torch.nan_to_num(recon_loss)
            
            if hasattr(self.sampler, "update_with_local_losses"):
                self.sampler.update_with_local_losses(t, recon_loss_batch)
            loss = {'recon_loss': recon_loss.mean()}
            if hasattr(self.model, 'aux_loss'):
                loss.update(self.model.aux_loss())

            if self.use_vel_loss:
                vel_loss = vel_loss.mean(dim=-1) * motion_mask[:, 1:]
                vel_loss = vel_loss * key_weights[:, 1:]
                vel_loss_batch = vel_loss.sum(dim=1) / motion_mask[:, 1:].sum(dim=1)
                vel_loss_frame = vel_loss.sum() / (motion_mask[:, 1:] * key_weights[:, 1:]).sum()
                if self.loss_reduction == "frame":
                    vel_loss = vel_loss_frame
                else:
                    vel_loss = vel_loss_batch
                vel_loss = torch.nan_to_num(vel_loss)
                loss.update({'vel_loss': vel_loss.mean()})
            
            # SMPL-X 3D keypoint loss
            if self.initialize_smplx_model:
                # Recover predicted 3D poses
                c_index_3d, c_index_6d = get_train_joint_idx()
                bs, ts, _ = pred.shape
                pred_3d = kwargs['origin_motion']
                recover_3d_pred = matrix_to_axis_angle(
                    rotation_6d_to_matrix(pred.reshape(bs * ts, -1, 6))
                    ).reshape(bs * ts, -1, 3).reshape(bs, ts, -1, 3).reshape(bs, ts, -1)
                pred_3d[..., c_index_3d.tolist()] = recover_3d_pred
                pred_3d = pred_3d.reshape(bs, ts, -1, 3)

                # Subsample 1/8 of timesteps for efficiency
                indices = np.arange(ts)
                mask_indices = np.random.choice(indices, size=(ts // 8), replace=False)
                pred_3d_sub = pred_3d[:, mask_indices]
                ts_sub = pred_3d_sub.shape[1]
                
                # Get predicted 3D keypoints via SMPL-X forward kinematics
                hand_joints_id = [i for i in range(25, 55)]
                body_joints_id = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

                body_model_outs = self.smplx_layer(
                    return_verts=False,
                    betas=kwargs['betas'].repeat(1, ts_sub, 1).reshape(bs * ts_sub, -1), 
                    jaw_pose=pred_3d_sub[:, :, 0:1].reshape(bs * ts_sub, -1, 3).reshape(bs * ts_sub, -1), 
                    leye_pose=pred_3d_sub[:, :, 1:2].reshape(bs * ts_sub, -1, 3).reshape(bs * ts_sub, -1), 
                    reye_pose=pred_3d_sub[:, :, 2:3].reshape(bs * ts_sub, -1, 3).reshape(bs * ts_sub, -1), 
                    global_orient=pred_3d_sub[:, :, 3:4].reshape(bs * ts_sub, -1, 3).reshape(bs * ts_sub, -1), 
                    body_pose=pred_3d_sub[:, :, 4:25].reshape(bs * ts_sub, -1, 3).reshape(bs * ts_sub, -1),
                    left_hand_pose=pred_3d_sub[:, :, 25:40].reshape(bs * ts_sub, -1, 3).reshape(bs * ts_sub, -1), 
                    right_hand_pose=pred_3d_sub[:, :, 40:55].reshape(bs * ts_sub, -1, 3).reshape(bs * ts_sub, -1),
                    expression=kwargs['expression'][:, mask_indices].reshape(bs * ts_sub, -1),
                    transl=kwargs['trans'][:, mask_indices].reshape(bs * ts_sub, -1)
                )
                # pred_vertices = body_model_outs.vertices.reshape(bs, ts, 10475, 3)
                pred_joints = body_model_outs.joints[:, self.smplx_model.joint_idx, :].reshape(
                    bs, ts_sub, len(self.smplx_model.joint_idx), 3)

                pred_joints += kwargs['trans'][:, mask_indices].unsqueeze(2)

                pred_joints = pred_joints[:, :, body_joints_id+hand_joints_id, :].reshape(bs, ts_sub, -1)

                # Compute loss against ground-truth 3D keypoints
                gt_kp3d = kwargs['gt_kp3d'][:, mask_indices]
                gt_joints = gt_kp3d[:, :, body_joints_id+hand_joints_id, :].reshape(bs, ts_sub, -1).detach()

                kp_3d_loss = self.kp_3d_criterion(pred_joints, gt_joints, reduction_override='none')
                motion_mask_sub = motion_mask[:, mask_indices]
                key_weights_sub = key_weights[:, mask_indices]
                kp_3d_loss = kp_3d_loss.mean(dim=-1) * motion_mask_sub
                kp_3d_loss = kp_3d_loss * key_weights_sub
                kp_3d_loss_batch = \
                    kp_3d_loss.sum(dim=1) / motion_mask_sub.sum(dim=1)
                kp_3d_loss_frame = \
                    kp_3d_loss.sum() / (motion_mask_sub * key_weights_sub).sum()
                if self.loss_reduction == "frame":
                    kp_3d_loss = kp_3d_loss_frame
                else:
                    kp_3d_loss = kp_3d_loss_batch
                kp_3d_loss = torch.nan_to_num(kp_3d_loss)
                loss.update({'kp_3d_loss': kp_3d_loss.mean()})
            return loss
        else:
            dim_pose = kwargs['motion'].shape[-1]
            model_kwargs = self.model.get_precompute_condition(
                device=motion.device, **kwargs)
            model_kwargs['motion_mask'] = motion_mask
            model_kwargs['motion_length'] = motion_length
            model_kwargs['num_intervals'] = num_intervals
            
            inference_kwargs = kwargs.get('inference_kwargs', {})
            if self.inference_type == 'ddpm':
                output = self.diffusion_test.p_sample_loop(
                    self.model, (B, T, dim_pose),
                    clip_denoised=False,
                    progress=False,
                    model_kwargs=model_kwargs,
                    **inference_kwargs)
            else:
                output = self.diffusion_test.ddim_sample_loop(
                    self.model, (B, T, dim_pose),
                    clip_denoised=False,
                    progress=False,
                    model_kwargs=model_kwargs,
                    eta=0,
                    **inference_kwargs)
            
            results = kwargs
            if getattr(self.model, "post_process") is not None:
                output = self.model.post_process(output)
            results['pred_motion'] = output
            results = self.split_results(results)
            return results
    
    def split_results(self, results):
        B = results['motion'].shape[0]
        output = []
        for i in range(B):
            batch_output = dict()
            batch_output['motion'] = to_cpu(results['motion'][i])
            batch_output['pred_motion'] = to_cpu(results['pred_motion'][i])
            batch_output['motion_length'] = to_cpu(results['motion_length'][i])
            batch_output['motion_mask'] = to_cpu(results['motion_mask'][i])
            if 'seed' in results.keys():
                batch_output['seed'] = to_cpu(results['seed'][i])
            if 'pred_motion_length' in results.keys():
                batch_output['pred_motion_length'] = \
                    to_cpu(results['pred_motion_length'][i])
            else:
                batch_output['pred_motion_length'] = \
                    to_cpu(results['motion_length'][i])
            if 'origin_motion' in results.keys():
                batch_output['origin_motion'] = to_cpu(results['origin_motion'][i])
            if 'pred_motion_mask' in results:
                batch_output['pred_motion_mask'] = \
                    to_cpu(results['pred_motion_mask'][i])
            else:
                batch_output['pred_motion_mask'] = \
                    to_cpu(results['motion_mask'][i])
            if 'motion_metas' in results.keys():
                motion_metas = results['motion_metas'][i]
                if 'origin_motion' in motion_metas.keys():
                    batch_output['origin_motion'] = motion_metas['origin_motion']
                if 'pkl_path' in motion_metas.keys():
                    batch_output['pkl_path'] = motion_metas['pkl_path']
                if 'wav_path' in motion_metas.keys():
                    batch_output['wav_path'] = motion_metas['wav_path']
                if 'motion_first_frame' in motion_metas.keys():
                    batch_output['motion_first_frame'] = motion_metas['motion_first_frame']
            output.append(batch_output)
        return output