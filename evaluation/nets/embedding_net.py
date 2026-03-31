"""
https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context.git
"""

import torch
import torch.nn as nn


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def ConvNormRelu(in_channels, out_channels, downsample=False, padding=0, batchnorm=True):
    if not downsample:
        k = 3
        s = 1
    else:
        k = 4
        s = 2

    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
    norm_block = nn.BatchNorm1d(out_channels)

    if batchnorm:
        net = nn.Sequential(
            conv_block,
            norm_block,
            nn.LeakyReLU(0.2, True)
        )
    else:
        net = nn.Sequential(
            conv_block,
            nn.LeakyReLU(0.2, True)
        )

    return net


class PoseEncoderConv(nn.Module):
    def __init__(self, length, dim):
        super().__init__()

        self.net = nn.Sequential(
            ConvNormRelu(dim, 32, batchnorm=True),
            ConvNormRelu(32, 64, batchnorm=True),
            ConvNormRelu(64, 64, True, batchnorm=True),
            nn.Conv1d(64, 32, 3)
        )

        self.out_net = nn.Sequential(
            nn.Linear(1280, 512),  # for 90 frames
            nn.BatchNorm1d(512),
            nn.LeakyReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(True),
            nn.Linear(256, 128),
        )

        self.fc_mu = nn.Linear(128, 128)
        self.fc_logvar = nn.Linear(128, 128)

    def forward(self, poses, variational_encoding):
        out = self.net(poses)
        out = out.flatten(1)
        out = self.out_net(out)

        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        if variational_encoding:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar


class PoseDecoderConv(nn.Module):
    def __init__(self, length, dim):
        super().__init__()

        feat_size = 128

        self.pre_net = nn.Sequential(
            nn.Linear(feat_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(True),
            nn.Linear(256, 720),
        )

        self.net = nn.Sequential(
            nn.ConvTranspose1d(8, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 32, 3),
            nn.Conv1d(32, dim, 3),
        )

    def forward(self, feat):
        out = self.pre_net(feat)
        out = out.view(feat.shape[0], 8, -1)
        out = self.net(out)
        return out


class EmbeddingNet(nn.Module):
    def __init__(self, pose_dim, n_frames):
        super().__init__()
        self.pose_encoder = PoseEncoderConv(n_frames, pose_dim)
        self.decoder = PoseDecoderConv(n_frames, pose_dim)

    def forward(self, poses, variational_encoding=False):
        poses_feat, _, _ = self.pose_encoder(poses, variational_encoding)
        out_poses = self.decoder(poses_feat)
        return poses_feat, out_poses

    def extract(self, x):
        self.pose_encoder.eval()
        feat, _, _ = self.pose_encoder(x, False)
        return feat.transpose(0, 1), x


class TrainWrapper:
    '''
    Wrapper for loading pretrained FGD embedding network and extracting features.
    '''

    def __init__(self, args):
        self.device = torch.device(args.gpu)
        self.generator = EmbeddingNet(264, 90).to(self.device)  # 44 joints x 6d = 264

    def load_state_dict(self, state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            sub_dict = OrderedDict()
            if v is not None:
                for k1, v1 in v.items():
                    name = k1.replace('module.', '')
                    sub_dict[name] = v1
            new_state_dict[k] = sub_dict
        state_dict = new_state_dict
        if 'generator' in state_dict:
            self.generator.load_state_dict(state_dict['generator'])
        else:
            self.generator.load_state_dict(state_dict)

    def extract(self, joints):
        self.generator.eval()
        with torch.no_grad():
            feat, x = self.generator.extract(joints)
        return feat.transpose(0, 1), x
