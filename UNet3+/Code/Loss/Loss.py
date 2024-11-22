import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
import torchvision

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
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
class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)

class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=1, iou_weight=1, ms_ssim_weight=1, dice_weight=1, smooth=1e-6, channel=3):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight
        self.ms_ssim_weight = ms_ssim_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ms_ssim = MSSSIM(window_size=11, size_average=True, channel=channel)
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

    
    def focal_loss(self, logits, targets, alpha=1, gamma=1.8, reduce=True):
        BCE_loss= F.binary_cross_entropy_with_logits(logits, targets, reduction='none')#self.bce_loss_fn(logits, targets)
        #print("BCE:",BCE_loss)
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1-pt)**gamma * BCE_loss
        if reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

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
    def bce_loss(self, logits, targets):
        # Use BCEWithLogitsLoss for numerical stability
        return self.bce_loss_fn(logits, targets)
    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets)*self.focal_weight 
        ms_ssim_loss = 1 - self.ms_ssim(torch.sigmoid(logits), targets) * self.ms_ssim_weight
        dice = self.dice_loss(logits, targets) * self.dice_weight
        iou= self.iou_loss(logits,targets) * self.iou_weight 
        #bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')


        # Combined loss
        total_loss = focal + ms_ssim_loss +iou + dice

        # 개별 손실 값 로깅을 위해 반환
        return total_loss, focal, ms_ssim_loss,iou, dice,


