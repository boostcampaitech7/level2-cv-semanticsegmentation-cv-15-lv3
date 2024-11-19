import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


# Gaussian Window 생성
def gaussian(window_size, sigma):
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    return gauss / gauss.sum()


# Window 생성 (캐싱 추가)
_window_cache = {}

def create_window(window_size, channel=1):
    global _window_cache
    key = (window_size, channel)
    if key not in _window_cache:
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        _window_cache[key] = window
    return _window_cache[key]


# SSIM 계산
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


# MS-SSIM 계산
def msssim(img1, img2, window_size=11, size_average=True, normalize=False):
    device = img1.device
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=device)
    levels = weights.size(0)
    mssim = []
    mcs = []

    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True)
        mssim.append(sim)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, kernel_size=2)
        img2 = F.avg_pool2d(img2, kernel_size=2)

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    output = torch.prod(mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1])
    return output


# MS-SSIM Module
class MSSSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)


# Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, smooth=1e-6, ms_ssim_window_size=11):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.ms_ssim = MSSSIM(window_size=ms_ssim_window_size, size_average=True, channel=1)

    def focal_loss(self, logits, targets, alpha=0.8, gamma=2.0):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-6, max=1-1e-6)
        focal_loss = -alpha * (1 - probs) ** gamma * targets * torch.log(probs) \
                     - (1 - alpha) * probs ** gamma * (1 - targets) * torch.log(1 - probs)
        return focal_loss.mean()

    def iou_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = torch.clamp(probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection, min=self.smooth)
        iou_loss = 1 - (intersection + self.smooth) / (union + self.smooth)
        return iou_loss.mean()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        focal = self.focal_loss(probs, targets)
        iou = self.iou_loss(probs, targets)
        ms_ssim_loss = 1 - self.ms_ssim(probs, targets)
        return self.alpha * focal + self.beta * iou + self.gamma * ms_ssim_loss
