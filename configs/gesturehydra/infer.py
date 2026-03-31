# dataset settings
crop_size = 150
min_length = 100
short_seg_num = 3
short_window = [10, 25]
long_window = [80, 120]
initialize_smplx_model = False
use_motion_style = True

num_workers = 24
batch_size = 128


if initialize_smplx_model:
    data_keys = ['motion', 'motion_mask', 'motion_length', 'wavlm', 'seed', 'key_frame', 'origin_motion', 'expression', 'betas', 'trans', 'gt_kp3d', 'motion_style', 'motion_style_mask']
else:
    data_keys = ['motion', 'motion_mask', 'motion_length', 'wavlm', 'seed', 'key_frame', 'origin_motion', 'expression', 'betas', 'trans', 'motion_style', 'motion_style_mask']
meta_keys = ['motion_first_frame']
train_pipeline = [
    dict(type='LoadPreloadCropAudioCondition', crop_size=crop_size, audio_aug=False),
    dict(type='GetSeedConditionWithKeyFrame', short_seg_num=short_seg_num, short_window=short_window, long_window=long_window),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=data_keys, meta_keys=meta_keys)
]

data = dict(
    use_webdataset=True,
    ntrain=100000,
    train=dict(type='SmplxA2GPreLoadDataloader',
        data_path="data/streamer-dataset/train/tars/{000000..000029}.tar",
        dataset_name='streamer',
        batch_size=batch_size,
        num_workers=num_workers,
        pipeline=train_pipeline,
        motion_dim=264,
        sequence_length=crop_size,

        min_length=min_length,
        initialize_smplx_model=initialize_smplx_model,
        use_motion_style=use_motion_style,
        csv_data_path='data/datasets/streamer/smplx_wav_va_train.csv',
    ),
    test=dict(
        type='SmplxA2GPreLoadDataloader',
        data_path='data/streamer-dataset/test_seen/tars/{000000..000009}.tar',
        dataset_name='streamer',
        batch_size=batch_size,
        num_workers=num_workers,
        pipeline=train_pipeline,
        motion_dim=264,
        sequence_length=crop_size,
        min_length=min_length,
        initialize_smplx_model=initialize_smplx_model,
        use_motion_style=use_motion_style,
        csv_data_path='data/datasets/streamer/smplx_wav_va_train.csv',
        test_mode=True))

# checkpoint saving
checkpoint_config = dict(interval=5000)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# optimizer
optimizer = dict(type='Adam', lr=1e-5)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[300000])
runner = dict(type='BF16IterBasedRunner', max_iters=500000)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

input_feats = 792//3
max_seq_len = 150
latent_dim = 384
time_embed_dim = 2048
audio_latent_dim = 1024
audio_feat_dim = 256
ff_size = 384
num_heads = 4
dropout = 0
dataset_name = "streamer"

# model settings
model = dict(type='MotionDiffusionA2G',
             model=dict(type='GestureHydraA2GTransformer',
                        input_feats=input_feats,
                        max_seq_len=max_seq_len,
                        latent_dim=latent_dim * 4,
                        time_embed_dim=time_embed_dim,
                        num_layers=8,
                        use_keyframe_mask=True,
                        use_motion_style=use_motion_style,
                        ca_block_cfg=dict(type='HybridModalityAttention',
                                          latent_dim=latent_dim,
                                          audio_latent_dim=audio_feat_dim,
                                          num_heads=num_heads,
                                          num_audio_heads=1,
                                          time_embed_dim=time_embed_dim,
                                          dropout=dropout,
                                          use_base_attention=True),
                        ffn_cfg=dict(latent_dim=latent_dim,
                                     ffn_dim=ff_size,
                                     dropout=dropout,
                                     time_embed_dim=time_embed_dim,
                                     num_heads=num_heads),
                        audio_encoder=dict(pretrained_model='checkpoints/chinese-wav2vec2-large-fairseq-ckpt/',
                                           audio_latent_dim=audio_latent_dim,
                                           audio_feat_dim=audio_feat_dim,
                                           num_layers=2),
                        pose_encoder_cfg=dict(dataset_name=dataset_name,
                                              latent_dim=latent_dim,
                                              input_dim=input_feats),
                        pose_decoder_cfg=dict(dataset_name=dataset_name,
                                              latent_dim=latent_dim,
                                              output_dim=input_feats),
                        scale_func_cfg=dict(scale=4.5),
                        use_pos_embedding=True),
             loss_recon=dict(type='MSELoss', loss_weight=10, reduction='none'),
             loss_kp_3d=dict(type='L1Loss', loss_weight=1, reduction='none'),
             loss_vel=dict(type='L1Loss', loss_weight=1, reduction='none'),
             diffusion_train=dict(
                 beta_scheduler='cosine',
                 diffusion_steps=1000,
                 model_mean_type='start_x',
                 model_var_type='fixed_large'),
             diffusion_test=dict(
                 beta_scheduler='cosine',
                 diffusion_steps=1000,
                 model_mean_type='start_x',
                 model_var_type='fixed_large',
                 respace='15,15,8,6,6',
             ),
             inference_type='ddim',
             loss_reduction='frame',
             initialize_smplx_model=initialize_smplx_model,
             use_vel_loss=True,
    )

bf16 = dict()
