# Copyright (C) 2025, TSGaussian
# TSGaussian research group, https://github.com/leon2000-ai/TSGaussian
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scipy.spatial import cKDTree

###
import math

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def masked_l1_loss(network_output, gt, mask):
    mask = mask.float()[None,:,:].repeat(gt.shape[0],1,1)
    loss = torch.abs((network_output - gt)) * mask
    loss = loss.sum() / mask.sum()
    return loss

def weighted_l1_loss(network_output, gt, weight):
    loss = torch.abs((network_output - gt)) * weight
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def loss_cls_3d(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
    and the KL divergence.
    
    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param predictions: Tensor of shape (N, C), where C is the number of classes.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.
    
    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        predictions = predictions[indices]


    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]
    sample_preds = predictions[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_features, features)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor predictions using indexing
    neighbor_preds = predictions[neighbor_indices_tensor]

    # Compute KL divergence
    kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
    loss = kl.sum(dim=-1).mean()

    # Normalize loss into [0, 1]
    num_classes = predictions.size(1)
    normalized_loss = loss / num_classes

    return lambda_val * normalized_loss

### 加入深度损失
def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def margin_l2_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask

def loss_depth_smoothness(depth, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs() * weight_x).sum() +
            ((depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())
    return loss

def patch_norm_mse_loss(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def patch_norm_mse_loss_global(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size), std = input.std().detach())
    target_patches = normalize(patchify(target, patch_size), std = target.std().detach())
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def pearson_depth_loss(depth_src, depth_target):
    #co = pearson(depth_src.reshape(-1), depth_target.reshape(-1))

    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    co = (src * target).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co

def pearson_depth_loss_0(depth_src, depth_target):

    src = normalize(depth_src)
    target = normalize(depth_target)
    co = (src * target).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co

def local_pearson_loss(depth_src, depth_target, patch_size):

    input_patches = normalize(patchify(depth_src, patch_size))
    target_patches = normalize(patchify(depth_target, patch_size))
    loss = pearson_depth_loss_0(input_patches, target_patches)
    # Randomly select patch, top left corner of the patch (x_0,y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w
    # num_box_h = math.floor(depth_src.shape[0]/box_p)
    # num_box_w = math.floor(depth_src.shape[1]/box_p)
    # max_h = depth_src.shape[0] - box_p
    # max_w = depth_src.shape[1] - box_p
    # _loss = torch.tensor(0.0,device='cuda')
    # n_corr = int(p_corr * num_box_h * num_box_w)
    # x_0 = torch.randint(0, max_h, size=(n_corr,), device = 'cuda')
    # y_0 = torch.randint(0, max_w, size=(n_corr,), device = 'cuda')
    # x_1 = x_0 + box_p
    # y_1 = y_0 + box_p
    # _loss = torch.tensor(0.0,device='cuda')
    # for i in range(len(x_0)):
    #     _loss += pearson_depth_loss_0(depth_src[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1), depth_target[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1))
    return loss
