'''
Warning: metrics are for reference only, may have limited significance
'''
import numpy as np
import torch


def LVD(gt_kps, pr_kps, symmetrical=False, weight=False):
    gt_kps = gt_kps.squeeze()
    pr_kps = pr_kps.squeeze()

    gt_velocity = (gt_kps[1:] - gt_kps[:-1]).norm(p=2, dim=-1)
    pr_velocity = (pr_kps[1:] - pr_kps[:-1]).norm(p=2, dim=-1)

    return (pr_velocity-gt_velocity).abs().sum(dim=-1).mean()
