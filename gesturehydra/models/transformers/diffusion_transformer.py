from abc import ABCMeta, abstractmethod

import torch
from mmcv.runner import BaseModule
from torch import nn

from gesturehydra.models.utils.misc import zero_module
from gesturehydra.models.utils.position_encoding import timestep_embedding
from gesturehydra.models.utils.stylization_block import StylizationBlock

from ..builder import build_attention


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb, **kwargs):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y


class DecoderLayer(nn.Module):

    def __init__(self, sa_block_cfg=None, ca_block_cfg=None, ffn_cfg=None):
        super().__init__()
        self.sa_block = build_attention(sa_block_cfg)
        self.ca_block = build_attention(ca_block_cfg)
        self.ffn = FFN(**ffn_cfg)

    def forward(self, **kwargs):
        if self.sa_block is not None:
            x = self.sa_block(**kwargs)
            kwargs.update({'x': x})
        if self.ca_block is not None:
            x = self.ca_block(**kwargs)
            kwargs.update({'x': x})
        if self.ffn is not None:
            x = self.ffn(**kwargs)
        return x


class DiffusionTransformer(BaseModule, metaclass=ABCMeta):

    def __init__(self,
                 input_feats,
                 max_seq_len=240,
                 latent_dim=512,
                 time_embed_dim=2048,
                 num_layers=8,
                 sa_block_cfg=None,
                 ca_block_cfg=None,
                 ffn_cfg=None,
                 use_pos_embedding=True,
                 use_residual_connection=False,
                 time_embedding_type='sinusoidal',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.input_feats = input_feats
        self.max_seq_len = max_seq_len
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        self.use_pos_embedding = use_pos_embedding
        
        if self.use_pos_embedding:
            self.sequence_embedding = nn.Parameter(torch.randn(max_seq_len, latent_dim))

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.time_embedding_type = time_embedding_type
        if time_embedding_type == 'learnable':
            self.time_tokens = nn.Embedding(1000, self.latent_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.build_temporal_blocks(sa_block_cfg, ca_block_cfg, ffn_cfg)

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))
        self.use_residual_connection = use_residual_connection

    def build_temporal_blocks(self, sa_block_cfg, ca_block_cfg, ffn_cfg):
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.temporal_decoder_blocks.append(
                DecoderLayer(sa_block_cfg=sa_block_cfg,
                             ca_block_cfg=ca_block_cfg,
                             ffn_cfg=ffn_cfg))

    @abstractmethod
    def get_precompute_condition(self, **kwargs):
        pass

    @abstractmethod
    def forward_train(self, h, src_mask, emb, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, h, src_mask, emb, **kwargs):
        pass

    def forward(self,
                motion,
                timesteps,
                motion_mask=None,
                motion_length=None,
                num_intervals=1,
                **kwargs):
        """
        motion: B, T, D
        """
        T = motion.shape[1]
        conditions = self.get_precompute_condition(device=motion.device,
                                                   **kwargs)
        if len(motion_mask.shape) == 2:
            src_mask = motion_mask.clone().unsqueeze(-1)
        else:
            src_mask = motion_mask.clone()

        if self.time_embedding_type == 'sinusoidal':
            emb = self.time_embed(
                timestep_embedding(timesteps, self.latent_dim))
        else:
            emb = self.time_embed(self.time_tokens(timesteps))  # (bs, 2048)

        # B, T, latent_dim
        h = self.joint_embed(motion)
        
        if self.use_pos_embedding:
            h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        if self.training:
            output = self.forward_train(h=h,
                                        src_mask=src_mask,
                                        emb=emb,
                                        timesteps=timesteps,
                                        motion_length=motion_length,
                                        num_intervals=num_intervals,
                                        **conditions)
        else:
            output = self.forward_test(h=h,
                                       src_mask=src_mask,
                                       emb=emb,
                                       timesteps=timesteps,
                                       motion_length=motion_length,
                                       num_intervals=num_intervals,
                                       **conditions)
        if self.use_residual_connection:
            output = motion + output
        return output
