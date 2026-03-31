import numpy as np
import torch
from scipy import linalg

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class EmbeddingSpaceEvaluator:
    def __init__(self, body_ae, vae, device):

        # init embed net
        self.body_ae = body_ae

        # storage
        self.body_real_feat = []
        self.body_gene_feat = []

    def reset(self):
        self.body_real_feat = []
        self.body_gene_feat = []

    def push_samples(self, generated_poses, real_poses):
        # convert poses to latent features
        generated_poses = generated_poses.float()
        real_poses = real_poses.float()

        real_feat, _ = self.body_ae.extract(real_poses)
        generated_feat, _ = self.body_ae.extract(generated_poses)
        real_feat = real_feat.squeeze()
        generated_feat = generated_feat.reshape(-1, generated_feat.shape[1])
        self.body_real_feat.append(real_feat.data.cpu().numpy())
        self.body_gene_feat.append(generated_feat.data.cpu().numpy())

    def get_scores(self, type):

        if type == 'bh':
            gene_list = self.body_gene_feat
            real_list = self.body_real_feat
        else:
            raise TypeError

        generated_feats = np.vstack(gene_list)
        real_feats = np.vstack(real_list)

        def frechet_distance(samples_A, samples_B):
            A_mu = np.mean(samples_A, axis=0)
            A_sigma = np.cov(samples_A, rowvar=False)
            B_mu = np.mean(samples_B, axis=0)
            B_sigma = np.cov(samples_B, rowvar=False)
            try:
                frechet_dist = self.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
            except ValueError:
                frechet_dist = 1e+10
            return frechet_dist

        ####################################################################
        # frechet distance
        frechet_dist = frechet_distance(generated_feats, real_feats)

        ####################################################################
        # distance between real and generated samples on the latent feature space
        dists = []
        for i in range(real_feats.shape[0]):
            d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
            dists.append(d)
        feat_dist = np.mean(dists)

        return frechet_dist, feat_dist

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        a = diff.dot(diff)
        b = np.trace(sigma1)
        c = np.trace(sigma2)
        d = tr_covmean

        return (a + b + c - 2 * d)
