import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
import torchvision
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from config import CLASSES

'''
def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        L = 1  # Assuming normalized images in [0, 1]

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=False)

'''


class MS_SSIM_Loss(nn.Module):
    def __init__(self, data_range=1.0, size_average=True, win_size=11, win_sigma=1.5, weights=None):
        """
        MS-SSIM Loss for PyTorch models.
        Args:
            data_range (float): Value range of input images. Default is 1.0 (normalized images).
            size_average (bool): If True, average the MS-SSIM values over all samples.
            win_size (int): Gaussian window size. Default is 11.
            win_sigma (float): Standard deviation of the Gaussian window. Default is 1.5.
            weights (list): Weights for different MS-SSIM levels. Default is None (uses preset weights).
        """
        super(MS_SSIM_Loss, self).__init__()
        self.ms_ssim = MS_SSIM(
            data_range=data_range,
            size_average=size_average,
            win_size=win_size,
            win_sigma=win_sigma,
            weights=weights,
            channel=len(CLASSES)
        )
    
    def forward(self, logits, targets):
        """
        Forward pass for the loss calculation.
        Args:
            logits (Tensor): Model outputs, typically raw scores (B, C, H, W).
            targets (Tensor): Ground truth images (B, C, H, W) normalized to [0, 1].
        Returns:
            Tensor: MS-SSIM loss value.
        """
        # Convert logits to probabilities using Sigmoid (for binary/multi-label tasks) or Softmax (multi-class tasks)
        probs = torch.sigmoid(logits)  # Use softmax if multi-class: torch.softmax(logits, dim=1)
        
        # Ensure targets are of the same dtype as probs
        targets = targets.type_as(probs)
        
        # Calculate MS-SSIM (higher values indicate better similarity)
        ms_ssim_val = self.ms_ssim(probs, targets)
        
        # Return 1 - MS-SSIM as the loss (lower MS-SSIM indicates higher loss)
        return 1 - ms_ssim_val


class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=1, iou_weight=1, ms_ssim_weight=1, dice_weight=1, gdl_weight=0, smooth=1e-6, boundary_weight=0,channel=3):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight
        self.ms_ssim_weight = ms_ssim_weight
        self.dice_weight = dice_weight
        self.gdl_weight = gdl_weight
        self.smooth = smooth
        self.boundary_weight = boundary_weight
        self.ms_ssim = MS_SSIM_Loss()
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')  # BCE loss with logits

    def adaptive_focal_loss(self, logits, targets, alpha=1, gamma_min=1.5, gamma_max=4.0, reduce=True):
        # Compute BCE loss
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')#self.bce_loss_fn(logits, targets)

        # Compute pt (predicted probability for true class)
        pt = torch.exp(-BCE_loss)

        # Dynamically adjust gamma based on pt
        gamma = gamma_min + (1 - pt) * (gamma_max - gamma_min)
        gamma = torch.clamp(gamma, gamma_min, gamma_max)  # Ensure gamma stays within [gamma_min, gamma_max]

        # Compute Focal Loss
        F_loss = alpha * (1 - pt) ** gamma * BCE_loss

        # Reduce loss if required
        if reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

    def focal_loss(self, logits, targets, alpha=1, gamma=1.5, reduce=True):
        BCE_loss= F.binary_cross_entropy_with_logits(logits, targets, reduction='none')#self.bce_loss_fn(logits, targets)
        #print("BCE:",BCE_loss)
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1-pt)**gamma * BCE_loss
        if reduce:
            return torch.mean(F_loss)
        else:
            return F_loss.sum() / logits.size(0)

    def iou_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
        iou_loss = 1 - (intersection + self.smooth) / (union + self.smooth)
        return iou_loss.mean()

    def dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        sum_probs = probs.sum(dim=(2, 3))
        sum_targets = targets.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (sum_probs + sum_targets + self.smooth)
        return 1 - dice.mean()

    def gdl_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        d_probs_dx = torch.abs(probs[:, :, :, :-1] - probs[:, :, :, 1:])
        d_targets_dx = torch.abs(targets[:, :, :, :-1] - targets[:, :, :, 1:])
        d_probs_dy = torch.abs(probs[:, :, :-1, :] - probs[:, :, 1:, :])
        d_targets_dy = torch.abs(targets[:, :, :-1, :] - targets[:, :, 1:, :])
        gdl_x = torch.abs(d_probs_dx - d_targets_dx).mean()
        gdl_y = torch.abs(d_probs_dy - d_targets_dy).mean()
        return gdl_x + gdl_y
    
    def boundary_loss(self, logits, targets):
        # 1. Prediction을 확률값으로 변환
        pred = torch.sigmoid(logits)
        # 2. Laplacian kernel을 사용하여 경계 검출
        laplacian_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], device=logits.device).float().view(1, 1, 3, 3)
        # 3. 예측값과 타겟의 경계 추출
        pred_boundary = F.conv2d(pred, laplacian_kernel, padding=1)
        target_boundary = F.conv2d(targets, laplacian_kernel, padding=1)
        # 4. 경계 부근 가중치 맵 생성 및 손실 계산
        weight_map = F.max_pool2d(torch.sigmoid(target_boundary), kernel_size=3, stride=1, padding=1)
        boundary_bce = F.binary_cross_entropy_with_logits(pred_boundary, target_boundary)
        # 5. 가중치가 적용된 최종 boundary loss 계산
        return (boundary_bce * (1 + 2 * weight_map)).mean()

    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets) * self.focal_weight
        dice = self.dice_loss(logits, targets) * self.dice_weight
        iou = self.iou_loss(logits, targets) * self.iou_weight
        #bce=self.bce_loss_fn(logits, targets)
        #gdl = self.gdl_loss(logits, targets) * self.gdl_weight
        ms_ssim = self.ms_ssim(logits, targets) * self.ms_ssim_weight
        # boundary = self.boundary_loss(logits, targets) * self.boundary_weight
        
        # Combined loss
        total_loss = focal + dice + iou + ms_ssim #gdl
        return total_loss, focal, iou, dice, ms_ssim #gdl


'''
class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, channel)

    def forward(self, img1, img2):
        # Ensure the images are normalized
        img1 = img1 / img1.max() if img1.max() > 1 else img1
        img2 = img2 / img2.max() if img2.max() > 1 else img2
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)
    
'''
