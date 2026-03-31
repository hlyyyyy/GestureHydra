import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random

from gesturehydra.models.utils.misc import zero_module, set_requires_grad
from ..builder import SUBMODULES, build_attention
from ..utils.stylization_block import StylizationBlock
from .diffusion_transformer import DiffusionTransformer
from gesturehydra.models.utils.position_encoding import timestep_embedding
from mmcv.runner import auto_fp16


def get_part_slice(idx_list, func):
    result = []
    for idx in idx_list:
        result.extend(func(idx))
    return result



class SFFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim, **kwargs):
        super().__init__()
        num_heads = kwargs.get('num_heads', 8)
        self.num_heads = num_heads
        self.linear1_list = nn.ModuleList()
        self.linear2_list = nn.ModuleList()
        for i in range(num_heads):
            self.linear1_list.append(nn.Linear(latent_dim, ffn_dim))
            self.linear2_list.append(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim * self.num_heads, time_embed_dim,
                                         dropout)
    
    def forward(self, x, emb, **kwargs):
        B, T, D = x.shape
        x = x.reshape(B, T, self.num_heads, -1)
        output = []
        for i in range(self.num_heads):
            feat = x[:, :, i].contiguous()
            feat = self.dropout(self.activation(self.linear1_list[i](feat)))
            feat = self.linear2_list[i](feat)
            output.append(feat)
        y = torch.cat(output, dim=-1)
        y = x.reshape(B, T, D) + self.proj_out(y, emb)
        return y


class DecoderLayer(nn.Module):

    def __init__(self, ca_block_cfg=None, ffn_cfg=None):
        super().__init__()
        self.ca_block = build_attention(ca_block_cfg)
        self.ffn = SFFN(**ffn_cfg)

    def forward(self, **kwargs):
        if self.ca_block is not None:
            x = self.ca_block(**kwargs)
            kwargs.update({'x': x})
        if self.ffn is not None:
            x_ffn = self.ffn(**kwargs)
        return x_ffn




def get_smplx_joint_slice(idx):
    result = [
        0 + idx * 6,
        0 + idx * 6 + 1,
        0 + idx * 6 + 2,
        0 + idx * 6 + 3,
        0 + idx * 6 + 4,
        0 + idx * 6 + 5,
    ]
    return result


class GestureEncoder(nn.Module):

    def __init__(self,
                 dataset_name="streamer",
                 latent_dim=64,
                 input_dim=263):
        super().__init__()
        if dataset_name == "streamer":
            func = get_smplx_joint_slice
            self.stem_slice = get_part_slice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], func)
            self.lhand_slice = get_part_slice([14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], func)
            self.rhand_slice = get_part_slice([29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], func)
            self.body_slice = get_part_slice([_ for _ in range(44)], func)
        else:
            raise ValueError()

        self.stem_embed = nn.Linear(len(self.stem_slice), latent_dim)
        self.lhand_embed = nn.Linear(len(self.lhand_slice), latent_dim)
        self.rhand_embed = nn.Linear(len(self.rhand_slice), latent_dim)
        self.body_embed = nn.Linear(len(self.body_slice), latent_dim)
        assert len(set(self.body_slice)) == input_dim

    def forward(self, motion):
        stem_feat = self.stem_embed(motion[:, :, self.stem_slice].contiguous())
        lhand_feat = self.lhand_embed(motion[:, :, self.lhand_slice].contiguous())
        rhand_feat = self.rhand_embed(motion[:, :, self.rhand_slice].contiguous())
        body_feat = self.body_embed(motion[:, :, self.body_slice].contiguous())
        feat = torch.cat((stem_feat, lhand_feat, rhand_feat, body_feat),
                         dim=-1)
        return feat
    

class GestureDecoder(nn.Module):

    def __init__(self,
                 dataset_name="streamer",
                 latent_dim=64,
                 output_dim=263):
        super().__init__()
        self.dataset_name = dataset_name
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        if dataset_name == "streamer":
            func = get_smplx_joint_slice
            self.stem_slice = get_part_slice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], func)
            self.lhand_slice = get_part_slice([14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], func)
            self.rhand_slice = get_part_slice([29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], func)
            self.body_slice = get_part_slice([_ for _ in range(44)], func)
        else:
            raise ValueError()

        self.stem_out = nn.Linear(latent_dim, len(self.stem_slice))
        self.lhand_out = nn.Linear(latent_dim, len(self.lhand_slice))
        self.rhand_out = nn.Linear(latent_dim, len(self.rhand_slice))
        self.body_out = nn.Linear(latent_dim, len(self.body_slice))

    def forward(self, motion):
        B, T = motion.shape[:2]
        D = self.latent_dim
        stem_feat = self.stem_out(motion[:, :, :D].contiguous())
        lhand_feat = self.lhand_out(motion[:, :, D:2 * D].contiguous())
        rhand_feat = self.rhand_out(motion[:, :, 2 * D:3 * D].contiguous())
        body_feat = self.body_out(motion[:, :, 3 * D:].contiguous())
        output = torch.zeros(B, T, self.output_dim).type_as(motion)
        output[:, :, self.stem_slice] = stem_feat.type_as(motion)
        output[:, :, self.lhand_slice] = lhand_feat.type_as(motion)
        output[:, :, self.rhand_slice] = rhand_feat.type_as(motion)
        output = (output + body_feat).type_as(motion) / 2.0
        return output


def shuffle_segments(tensor, segment_length):
    if tensor.dim() != 3:
        raise ValueError("Input tensor must be 3-dimensional")
    C, T, H = tensor.size()
    num_segments = T // segment_length
    shuffled_tensor = tensor.clone()
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        shuffled_indices = torch.randperm(segment_length) + start
        shuffled_tensor[:, start:end, :] = tensor[:, shuffled_indices, :]
    return shuffled_tensor


@SUBMODULES.register_module()
class GestureHydraA2GTransformer(DiffusionTransformer):

    def __init__(self,
                 audio_encoder=None,
                 scale_func_cfg=None,
                 pose_encoder_cfg=None,
                 pose_decoder_cfg=None,
                 decouple=False,
                 use_motion_style=False,
                 use_keyframe_mask=False,
                 use_betas_style=False,
                 **kwargs):
        self.decouple = decouple
        self.use_motion_style = use_motion_style
        self.use_keyframe_mask = use_keyframe_mask
        self.use_betas_style = use_betas_style
        super().__init__(**kwargs)
        self.scale_func_cfg = scale_func_cfg
        self.joint_embed = GestureEncoder(**pose_encoder_cfg)
        self.out = zero_module(GestureDecoder(**pose_decoder_cfg))
        
        # for audio extractor
        if audio_encoder is not None:
            init_wavmodel = audio_encoder.get('init_wav_model', False)
            model_path = audio_encoder['pretrained_model']
            audio_latent_dim = audio_encoder['audio_latent_dim']
            audio_feat_dim = audio_encoder['audio_feat_dim']
            if init_wavmodel:
                from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
                self.audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
                self.wav_model = Wav2Vec2Model.from_pretrained(model_path)
                self.wav_model.eval()
                set_requires_grad(self.wav_model, requires_grad=False)
            
            self.audio_feature_map = nn.Sequential(
                nn.Linear(audio_latent_dim, audio_feat_dim),
                nn.SiLU(),
                nn.Linear(audio_feat_dim, audio_feat_dim),
            )

            num_audio_layers = audio_encoder.get('num_layers', 0)
            if num_audio_layers > 0:
                self.use_audio_finetune = True
                audioTransEncoderLayer = nn.TransformerEncoderLayer(
                    d_model=audio_feat_dim,
                    nhead=4,
                    dim_feedforward=2048,
                    dropout=0,
                    activation='gelu')
                self.audioTransEncoder = nn.TransformerEncoder(
                    audioTransEncoderLayer, num_layers=num_audio_layers)
                self.audio_ln = nn.LayerNorm(audio_feat_dim)
            else:
                self.use_audio_finetune = False
            
            # Extract seed features (guiding motion)
            if not use_keyframe_mask:
                self.seed_emb = nn.Sequential(
                    nn.Linear(self.input_feats, audio_feat_dim, bias=False),
                    nn.SiLU(),
                    nn.Linear(audio_feat_dim, audio_feat_dim, bias=False),
                )
            else:
                self.seed_emb = nn.Sequential(
                    nn.Linear(self.input_feats+1, audio_feat_dim, bias=False),
                    nn.SiLU(),
                    nn.Linear(audio_feat_dim, audio_feat_dim, bias=False),
                )
            
            
            # extract style feature
            if self.use_motion_style:
                self.style_joint_embed = nn.Linear(self.input_feats, audio_feat_dim)
                self.global_motion_token = nn.Parameter(
                    torch.randn(audio_feat_dim * 2, audio_feat_dim))
                styleTransEncoderLayer = nn.TransformerEncoderLayer(
                    d_model=audio_feat_dim,
                    nhead=4,
                    dim_feedforward=1024,
                    dropout=0,
                    activation='gelu')
                self.style_pos_emb = nn.Parameter(torch.randn(1000, audio_feat_dim))
                self.styleTransEncoder = nn.TransformerEncoder(
                    styleTransEncoderLayer, num_layers=2)
                self.style_norm = nn.LayerNorm(audio_feat_dim)
                self.style_cond_proj = nn.Linear(audio_feat_dim, self.time_embed_dim)
            
            if self.use_betas_style:
                self.style_emb = nn.Linear(10, self.time_embed_dim)
            
        self.fp16_enabled = False

    def build_temporal_blocks(self, sa_block_cfg, ca_block_cfg, ffn_cfg):
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            if isinstance(ffn_cfg, list):
                ffn_cfg_block = ffn_cfg[i]
            else:
                ffn_cfg_block = ffn_cfg
            self.temporal_decoder_blocks.append(
                DecoderLayer(ca_block_cfg=ca_block_cfg, ffn_cfg=ffn_cfg_block))

    def scale_func(self, timestep):
        if not self.decouple:
            scale = self.scale_func_cfg['scale']
            w = (1 - (1000 - timestep) / 1000) * scale + 1
            output = {'audio_coef': w, 'none_coef': 1 - w}
        else:
            coarse_scale = self.scale_func_cfg['coarse_scale']
            w = (1 - (1000 - timestep) / 1000) * coarse_scale + 1
            if timestep > 100:
                if random.randint(0, 1) == 0:
                    output = {
                        'both_coef': w,
                        'audio_coef': 1 - w,
                        'retr_coef': 0,
                        'none_coef': 0
                    }
                else:
                    output = {
                        'both_coef': 0,
                        'audio_coef': 0,
                        'retr_coef': w,
                        'none_coef': 1 - w
                    }
            else:
                both_coef = self.scale_func_cfg['both_coef']
                audio_coef = self.scale_func_cfg['audio_coef']
                retr_coef = self.scale_func_cfg['retr_coef']
                none_coef = 1 - both_coef - audio_coef - retr_coef
                output = {
                    'both_coef': both_coef,
                    'audio_coef': audio_coef,
                    'retr_coef': retr_coef,
                    'none_coef': none_coef
                }
        return output

    def aux_loss(self):
        return {}
    
    def wav2feat_cn(self, wav_input_16khz, device=torch.device("cuda")):
        with torch.no_grad():
            input_values = self.audio_feature_extractor(wav_input_16khz, sampling_rate=16000, return_tensors="pt").input_values.to(device)
            
            chunk_len = 16000 * 15
            wav_len = input_values.shape[-1]   # [1, len]
            num_chunks = wav_len // chunk_len + 1
            input_values = torch.nn.functional.pad(input_values, (0, chunk_len * num_chunks - wav_len))
            input_values = input_values.reshape(num_chunks, chunk_len)
            
            # save cuda memory
            rep = []
            for i in range(0, num_chunks, 10):   # 10: batch
                rep.append(self.wav_model(input_values[i:i + 10]).last_hidden_state[0].cpu())
            rep = torch.cat(rep, dim=0)
            
            del input_values
            return rep

    def get_precompute_condition(self,
                                 motion_length=None,
                                 xf_audio=None,
                                 xf_style=None,
                                 xf_style_motion=None,
                                 xf_keyf=None,
                                 device=None,
                                 **kwargs):
        if xf_audio is None:
            if 'wavlm' in kwargs.keys():
                wavlm = kwargs['wavlm']  # [bs, n_seq, 1133]
            else:
                wav_feat_1, wav, wav_feat_3 = kwargs['wav_feat_1'], kwargs['wav'], kwargs['wav_feat_3']
                
                # wav: [bs, 153600]
                # wav_feat_1: [bs, 240, 108]
                # wav_feat_3: [bs, 240, 1]
                wavlm_f_list = []
                for idx in range(wav.shape[0]):
                    wav_idx = wav[idx]
                    wavlm_f = self.wav2feat_cn(wav_idx, device)
                    wavlm_f = F.interpolate(wavlm_f.unsqueeze(0).transpose(1, 2), size=wav_feat_1.shape[1], align_corners=True, mode='linear').transpose(1, 2).squeeze()
                    wavlm_f_list.append(wavlm_f)
                
                # embed wav
                wavlm_f = torch.stack(wavlm_f_list, dim=0).to(device)  # (bs, 240, 1024)
                wavlm = torch.cat([wavlm_f, wav_feat_1, wav_feat_3], dim=-1)    # (bs, 240, 1133)

            if self.use_motion_style:
                motion_style = kwargs['motion_style']
                motion_style_mask = kwargs['motion_style_mask']
            
            if kwargs.get('key_frame', None) is not None:
                key_frame = kwargs['key_frame']
                seed = kwargs['seed'].to(torch.float32)
                wavlm = self.audio_feature_map(wavlm.to(torch.float32))
                if self.use_audio_finetune:
                    wavlm = self.audioTransEncoder(wavlm.permute(1, 0, 2))
                    wavlm = self.audio_ln(wavlm)
                    wavlm = wavlm.permute(1, 0, 2)
                cond_audio_full = wavlm.to(torch.float32).to(device, non_blocking=True)  # [10, 215, 256]
                # embed seed
                if self.use_keyframe_mask:
                    key_frame = key_frame.to(seed.dtype)
                    seed = torch.cat([seed, key_frame.unsqueeze(-1)], dim=-1)
                cond_seed = self.seed_emb(seed)  # [10, 25, 256]

                cond_seed_audio = cond_audio_full
                # Replace cond_cls with seed embedding
                cond_cls_audio = cond_seed
            
            # style embedding
            if self.use_motion_style:
                bs = motion_length.shape[0]
                if self.training:
                    motion_style = shuffle_segments(motion_style, 25)
                motion_style = self.style_joint_embed(motion_style.float())  # [8, 240, 256]
                motion_style = motion_style.permute(1, 0, 2)  # [240, bs, 256]
                # Each batch has its own set of tokens
                dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))  # [512, 8, 256]
                # create a bigger mask, to allow attend to emb
                dist_masks = torch.ones((bs, dist.shape[0]), dtype=bool, device=motion_style.device)  # [8, 512]
                aug_mask = torch.cat((dist_masks, motion_style_mask), 1).to(torch.bool)  # [8, 752]
                # adding the embedding token for all sequences
                motion_style = torch.cat((dist, motion_style), 0)  # [752, 8, 256]
                motion_style = motion_style + self.style_pos_emb.unsqueeze(1)[:motion_style.shape[0], :, :]  # # [752, 8, 256]
                dist = self.styleTransEncoder(motion_style, src_key_padding_mask=~aug_mask)  # [752, 8, 256]
                dist = self.style_norm(dist)
                cond_style_motion = self.style_cond_proj(dist[0])

            # betas style embedding
            if self.use_betas_style:
                style = kwargs['betas'][:, 0, :]
                cond_style = style.to(device)
                cond_style = self.style_emb(cond_style)
                xf_style = cond_style

            if self.use_motion_style:
                xf_style_motion = cond_style_motion

            xf_audio = cond_seed_audio
            xf_keyf = cond_cls_audio
        
        output = {'xf_audio': xf_audio, 'xf_style': xf_style, 'xf_keyf': xf_keyf}
        if self.use_betas_style:
            output['xf_style'] = xf_style
        if self.use_motion_style:
            output['xf_style_motion'] = xf_style_motion
        return output

    def post_process(self, motion):
        return motion
    
    @auto_fp16(apply_to=('motion'))
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
                                                   motion_length=motion_length,
                                                   motion_mask=motion_mask,
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
        
        # style add to time embedding
        if self.use_betas_style:
            emb = emb + conditions['xf_style']
        if self.use_motion_style:
            emb = emb + conditions['xf_style_motion']
        
        # B, T, latent_dim
        h = self.joint_embed(motion)
        
        if self.use_pos_embedding:
            h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]
        
        if kwargs.get('key_frame', None) is not None:
            key_frame = kwargs['key_frame']
            conditions['key_frame'] = key_frame

        if self.training:
            output = self.forward_train(h=h,
                                        src_mask=src_mask,
                                        emb=emb,
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

    def forward_train(self,
                      h=None,
                      src_mask=None,
                      emb=None,
                      xf_audio=None,
                      motion_length=None,
                      num_intervals=1,
                      **kwargs):
        B, T = h.shape[0], h.shape[1]
        cond_type = torch.randint(0, 100, size=(B, 1, 1)).to(h.device)
        for module in self.temporal_decoder_blocks:
            h = module(x=h,
                    xf_audio=xf_audio,
                    emb=emb,
                    src_mask=src_mask,
                    cond_type=cond_type,
                    motion_length=motion_length,
                    num_intervals=num_intervals,
                    **kwargs)

        output = self.out(h).view(B, T, -1).contiguous()
        return output

    def forward_test(self,
                     h=None,
                     src_mask=None,
                     emb=None,
                     xf_audio=None,
                     timesteps=None,
                     motion_length=None,
                     num_intervals=1,
                     **kwargs):
        if not self.decouple:
            B, T = h.shape[0], h.shape[1]
            audio_cond_type = torch.zeros(B, 1, 1).to(h.device) + 1
            none_cond_type = torch.zeros(B, 1, 1).to(h.device)

            all_cond_type = torch.cat((audio_cond_type, none_cond_type), dim=0)
            h = h.repeat(2, 1, 1)
            xf_audio = xf_audio.repeat(2, 1, 1)
            kwargs['xf_keyf'] = kwargs['xf_keyf'].repeat(2, 1, 1)
            emb = emb.repeat(2, 1)
            src_mask = src_mask.repeat(2, 1, 1)
            motion_length = motion_length.repeat(2, 1)
            for module in self.temporal_decoder_blocks:
                h = module(x=h,
                           xf_audio=xf_audio,
                           emb=emb,
                           src_mask=src_mask,
                           cond_type=all_cond_type,
                           motion_length=motion_length,
                           num_intervals=num_intervals,
                           **kwargs)
            out = self.out(h).view(2 * B, T, -1).contiguous()
            out_audio = out[:B].contiguous()
            out_none = out[B:].contiguous()

            coef_cfg = self.scale_func(int(timesteps[0]))
            audio_coef = coef_cfg['audio_coef']
            none_coef = coef_cfg['none_coef']
            output = out_audio * audio_coef + out_none * none_coef
        else:
            B, T = h.shape[0], h.shape[1]
            both_cond_type = torch.zeros(B, 1, 1).to(h.device) + 99
            audio_cond_type = torch.zeros(B, 1, 1).to(h.device) + 1
            retr_cond_type = torch.zeros(B, 1, 1).to(h.device) + 10
            none_cond_type = torch.zeros(B, 1, 1).to(h.device)

            all_cond_type = torch.cat(
                (both_cond_type, audio_cond_type, retr_cond_type, none_cond_type),
                dim=0)
            h = h.repeat(4, 1, 1)
            xf_audio = xf_audio.repeat(4, 1, 1)
            kwargs['xf_keyf'] = kwargs['xf_keyf'].repeat(4, 1, 1)
            emb = emb.repeat(4, 1)
            src_mask = src_mask.repeat(4, 1, 1)
            motion_length = motion_length.repeat(4, 1)

            for module in self.temporal_decoder_blocks:
                h = module(x=h,
                        xf_audio=xf_audio,
                        emb=emb,
                        src_mask=src_mask,
                        cond_type=all_cond_type,
                        motion_length=motion_length,
                        num_intervals=num_intervals,
                        **kwargs)
            out = self.out(h).view(4 * B, T, -1).contiguous()
            out_both = out[:B].contiguous()
            out_audio = out[B:2 * B].contiguous()
            out_retr = out[2 * B:3 * B].contiguous()
            out_none = out[3 * B:].contiguous()

            coef_cfg = self.scale_func(int(timesteps[0]))
            both_coef = coef_cfg['both_coef']
            audio_coef = coef_cfg['audio_coef']
            retr_coef = coef_cfg['retr_coef']
            none_coef = coef_cfg['none_coef']
            output = out_both * both_coef
            output += out_audio * audio_coef
            output += out_retr * retr_coef
            output += out_none * none_coef
        return output

