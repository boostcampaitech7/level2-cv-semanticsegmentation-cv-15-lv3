import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM
from config.config import Config

class Loss:
    @staticmethod
    def get_criterion(loss_type="bce"):
        """
        Get the loss criterion
        
        Args:
            loss_type (str): Type of loss to use
                - "bce": Binary Cross Entropy
                - "dice": Dice Loss
                - "focal": Focal Loss
                - "iou": IoU Loss
        """
        if loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        elif loss_type == "dice":
            return DiceLoss()
        elif loss_type == "focal":
            return FocalLoss()
        elif loss_type == "iou":
            return IoULoss()
        elif loss_type == "bie":
            print("Hybrid Loss Training")
            return BIELoss()
        elif loss_type == "hybrid":
            print("Hybrid Loss Training")
            return HybridLoss()
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented")

# 추가적인 loss 함수들을 정의할 수 있습니다
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Dice Loss 구현

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        
    def forward(self, logits, targets, alpha=1, gamma=1.5, reduce=True):
        """
        기본 Focal Loss
        Args:
            logits: 모델 예측값 
            targets: 정답 레이블
            alpha: 가중치 계수 (default: 1)
            gamma: focusing 파라미터 (default: 1.5)
            reduce: 손실 값 평균 계산 여부 (default: True)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1-pt)**gamma * BCE_loss
        
        if reduce:
            return torch.mean(F_loss)
        else:
            return F_loss.sum() / logits.size(0)

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma_min=1.5, gamma_max=4.0, reduction='mean'):
        """
        Adaptive Focal Loss for PyTorch models.
        Args:
            alpha (float): Weight factor for the loss. Default is 1.0
            gamma_min (float): Minimum value for the adaptive gamma. Default is 1.5
            gamma_max (float): Maximum value for the adaptive gamma. Default is 4.0
            reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.reduction = reduction

    def forward(self, logits, targets, alpha=1, gamma_min=1.5, gamma_max=4.0, reduce=True):
        """
        Adaptive Focal Loss - pt값에 따라 gamma를 동적으로 조절
        Args:
            logits: 모델 예측값
            targets: 정답 레이블 
            alpha: 가중치 계수 (default: 1)
            gamma_min: 최소 gamma 값 (default: 1.5)
            gamma_max: 최대 gamma 값 (default: 4.0)
            reduce: 손실 값 평균 계산 여부 (default: True)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        
        # pt값에 따라 gamma 동적 조절
        gamma = gamma_min + (1 - pt) * (gamma_max - gamma_min)
        gamma = torch.clamp(gamma, gamma_min, gamma_max)
        
        F_loss = alpha * (1 - pt) ** gamma * BCE_loss
        
        if reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        intersection = (pred * target).sum(dim=(2, 3))
        total = (pred + target).sum(dim=(2, 3))
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()

class LogIoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        intersection = (pred * target).sum(dim=(2, 3))
        total = (pred + target).sum(dim=(2, 3))
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return -torch.log(iou).mean()  # Log-IOU Loss

class BIELoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iou_loss = LogIoULoss()
        self.alpha = alpha  # BCE weight
        self.beta = beta   # IoU weight
        self.gamma = gamma # Edge weight
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        iou = self.iou_loss(pred, target)
        
        total_loss = (self.alpha * bce + 
                     self.beta * iou)
        
        return total_loss, bce, iou 
    
class MS_SSIM_Loss(nn.Module):
    def __init__(self, alpha=1.0, reduction='mean', win_size=11, win_sigma=1.5, data_range=1.0):
        """
        Multi-Scale Structural Similarity (MS-SSIM) Loss for PyTorch models.
        Args:
            alpha (float): Weight factor for the loss. Default is 1.0.
            reduction (str): Specifies the reduction to apply to the output.
                           Options: 'none' | 'mean' | 'sum'. Default: 'mean'.
            win_size (int): Window size for SSIM calculation. Default is 11.
            win_sigma (float): Standard deviation for Gaussian window. Default is 1.5.
            data_range (float): Range of the input data. Default is 1.0 for normalized images.
        """
        super(MS_SSIM_Loss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.ms_ssim = MS_SSIM(
            data_range=data_range,
            size_average=(reduction == 'mean'),
            win_size=win_size,
            win_sigma=win_sigma,
            channel=len(Config.CLASSES)
        )

    def forward(self, logits, targets):
        """
        Forward pass for the loss calculation.
        Args:
            logits (Tensor): Model outputs, typically raw scores (B, C, H, W).
            targets (Tensor): Ground truth masks (B, C, H, W) with values in {0, 1}.
        Returns:
            Tensor: MS-SSIM loss value.
        """
        # Convert logits to probabilities using Sigmoid
        probs = torch.sigmoid(logits)
        
        # Ensure targets are of the same dtype as probs
        targets = targets.type_as(probs)
        
        # Calculate MS-SSIM (higher values indicate better similarity)
        ms_ssim_val = self.ms_ssim(probs, targets)
        
        # Calculate loss (1 - MS-SSIM)
        loss = self.alpha * (1 - ms_ssim_val)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return loss.sum() / logits.size(0)
        else:
            return loss

class HybridLoss(nn.Module):
    def __init__(self, 
                 focal_alpha=1.0,
                 focal_gamma_min=1.5, 
                 focal_gamma_max=4.0,
                 ssim_alpha=1.0,
                 ssim_win_size=11,
                 ssim_win_sigma=1.5,
                 weights={'focal': 1.0, 'iou': 1.0, 'ssim': 1.0},
                 reduction='mean'):
        """
        Hybrid Loss combining Adaptive Focal Loss, Log-IoU Loss and MS-SSIM Loss
        Args:
            focal_alpha (float): Alpha parameter for Focal Loss
            focal_gamma_min (float): Minimum gamma for Adaptive Focal Loss
            focal_gamma_max (float): Maximum gamma for Adaptive Focal Loss
            ssim_alpha (float): Alpha parameter for MS-SSIM Loss
            ssim_win_size (int): Window size for MS-SSIM calculation
            ssim_win_sigma (float): Sigma for MS-SSIM calculation
            weights (dict): Weights for each loss component
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super(HybridLoss, self).__init__()
        
        self.focal_loss = AdaptiveFocalLoss(
            alpha=focal_alpha,
            gamma_min=focal_gamma_min,
            gamma_max=focal_gamma_max,
            reduction=reduction
        )
        
        self.iou_loss = LogIoULoss(
            reduction=reduction
        )
        
        self.ssim_loss = MS_SSIM_Loss(
            alpha=ssim_alpha,
            win_size=ssim_win_size,
            win_sigma=ssim_win_sigma,
            reduction=reduction
        )
        
        self.weights = weights
        self.reduction = reduction

    def forward(self, logits, targets):
        focal_loss = self.focal_loss(logits, targets)
        iou_loss = self.iou_loss(logits, targets)
        ssim_loss = self.ssim_loss(logits, targets)
        
        # Combine losses with weights
        total_loss = (
            self.weights['focal'] * focal_loss + 
            self.weights['iou'] * iou_loss + 
            self.weights['ssim'] * ssim_loss
        )
        
        return total_loss, focal_loss, iou_loss, ssim_loss

