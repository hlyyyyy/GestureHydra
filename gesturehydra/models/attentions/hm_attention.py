import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from ..builder import ATTENTIONS
from ..utils.stylization_block import StylizationBlock


@ATTENTIONS.register_module()
class HybridModalityAttention(nn.Module):
    def __init__(self, latent_dim, audio_latent_dim, num_heads, num_audio_heads,
                 time_embed_dim,
                 dropout, use_base_attention=False,
                 **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_audio_heads = num_audio_heads

        self.norm = nn.LayerNorm(latent_dim)
        self.audio_norm = nn.LayerNorm(audio_latent_dim)
        self.class_norm = nn.LayerNorm(latent_dim)
        
        self.seed_proj = nn.Linear(audio_latent_dim, num_heads * latent_dim, bias=False)
        self.audio_qkv = nn.Linear(audio_latent_dim, latent_dim * 2)
        self.motion_qkv = nn.Linear(latent_dim, latent_dim * 3)

        self.body_weight = nn.Parameter(torch.randn(num_heads, num_heads))
        
        # fusion block
        self.fusion_norm = nn.LayerNorm(3 * latent_dim)
        self.fusion_block = nn.Sequential(
            nn.Linear(3 * latent_dim, 3 * latent_dim),
            nn.SiLU(),
            nn.Linear(3 * latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.proj_out = StylizationBlock(latent_dim * num_heads,
                                         time_embed_dim, dropout)

        self.use_base_attention = use_base_attention
        self.fp16_enabled = False

    @auto_fp16(apply_to=('x', 'xf_audio', 'emb'))
    def forward(self, x, xf_audio, emb, src_mask, cond_type,
                **kwargs):
        """
        x: B, T, D (noised motion)
        xf_audio: B, N, P (audio features)
        """
        B, T, D = x.shape
        N = x.shape[1]
        H = self.num_heads
        L = self.latent_dim

        x = x.reshape(B, T, H, -1)
        xf_keyf = kwargs['xf_keyf']
        
        audio_feat = xf_audio.reshape(B, xf_audio.shape[1], self.num_audio_heads, -1)
        audio_feat = self.audio_qkv(self.audio_norm(audio_feat))
        
        seed_feat = xf_keyf.reshape(B, xf_keyf.shape[1], self.num_audio_heads, -1)
        seed_feat = self.seed_proj(seed_feat).reshape(B, T, H, -1)
        
        audio_cond_type = (cond_type % 10 > 0).float().unsqueeze(-1)
        audio_feat = (audio_feat * audio_cond_type).repeat(1, 1, H, 1)  # [8, 150, 6, 128]
        
        # residual connection
        fusion_feat = self.fusion_block(self.fusion_norm(torch.cat((x, audio_feat), dim=-1)))
        x += fusion_feat
        
        motion_feat = self.motion_qkv(self.norm(x) + self.class_norm(seed_feat))

        body_weight = F.softmax(self.body_weight, dim=1)
        body_value = motion_feat[:, :, :, :L]
        body_feat = torch.einsum('hl,bnld->bnhd', body_weight, body_value)
        body_feat = body_feat.reshape(B, T, D)

        if not self.use_base_attention:
            # B, N, D
            src_mask = src_mask.view(B, T, 1, 1)
            key_motion = motion_feat[:, :, :, L:2 * L].contiguous()
            key_motion = key_motion + (1 - src_mask) * -1000000
            key = key_motion
            key = F.softmax(key.view(B, N, H, -1), dim=1)

            value_motion = motion_feat[:, :, :, 2 * L:].contiguous() * src_mask
            value = value_motion
        else:
            mask_motion = src_mask.view(B, 1, T, 1)  # (bs, 1, seq_len, 1)
            key_motion = motion_feat[:, :, :, L:2 * L].contiguous()
            key = key_motion

            value_motion = motion_feat[:, :, :, 2 * L:].contiguous() * src_mask.unsqueeze(-1)
            value = value_motion
            mask = mask_motion  # (bs, 1, N, 1)

        if not self.use_base_attention:
            attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
            y = torch.einsum('bnhd,bhdl->bnhl', body_feat.reshape(B, T, H, -1), attention)
            y = x.reshape(B, T, D) + self.proj_out(y.reshape(B, T, D), emb)
        else:
            attention = torch.einsum('bnhl,bmhl->bnmh', body_feat.reshape(B, T, H, -1), key)  # (bs, T, N, H)
            attention = attention + (1 - mask) * -1000000  # (bs, T, N, H)
            attention = F.softmax(attention, dim=2)
            y = torch.einsum('bnmh,bmhl->bnhl', attention, value)  # (bs, T, H, -1)
            y = x.reshape(B, T, D) + self.proj_out(y.reshape(B, T, D), emb)
        return y
