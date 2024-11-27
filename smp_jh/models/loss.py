import torch.nn as nn
import torch
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
        elif loss_type == "hybrid":
            print("Hybrid Loss Training")
            return CombinedLoss()
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented")

# 추가적인 loss 함수들을 정의할 수 있습니다
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Dice Loss 구현

# class FocalLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Focal Loss 구현

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss

        Args:
            alpha (float): Balancing factor for positive/negative classes
            gamma (float): Focusing parameter to down-weight easy examples
            reduction (str): 'mean', 'sum', or 'none' for reduction type
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # BCE Loss 계산
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # Sigmoid로 확률값 계산
        inputs = torch.sigmoid(inputs)
        # Focal weight 계산
        focal_weight = (1 - inputs) ** self.gamma * targets + \
                       inputs ** self.gamma * (1 - targets)
        focal_loss = self.alpha * focal_weight * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
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

# class HybridLoss(nn.Module):
#     def __init__(self, alpha=0.5):
#         super().__init__()
#         self.bce_loss = nn.BCEWithLogitsLoss()
#         self.iou_loss = LogIoULoss()
#         self.alpha = alpha  # BCE와 IoU loss의 가중치 비율
    
#     def forward(self, pred, target):
#         bce = self.bce_loss(pred, target)
#         iou = self.iou_loss(pred, target)
#         return self.alpha * bce + (1 - self.alpha) * iou, bce, iou

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
            channel=len(Config.CLASSES)
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

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.iou_loss = IoULoss()
        self.ms_ssim_loss = MS_SSIM_Loss()
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        ms_ssim = self.ms_ssim_loss(pred, target)
        iou =  self.iou_loss(pred, target)
        hybrid = focal + ms_ssim + iou

        return hybrid, focal, ms_ssim, iou

class HybridLossLog(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.iou_loss = LogIoULoss()
        self.ms_ssim_loss = MS_SSIM_Loss()
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        ms_ssim = self.ms_ssim_loss(pred, target)
        iou =  self.iou_loss(pred, target)
        hybrid = focal + ms_ssim + iou

        return hybrid, focal, ms_ssim, iou

class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=1, iou_weight=1, ms_ssim_weight=1, dice_weight=0, gdl_weight=0, smooth=1e-6, boundary_weight=0,channel=3):
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
        # Laplacian kernel 정의
        laplacian_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], device=logits.device, dtype=torch.float32).reshape(1, 1, 3, 3)
        
        # 배치 크기와 채널 수에 맞게 커널 확장
        batch_size, channels = logits.shape[:2]
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        
        # 경계 추출
        pred_boundary = F.conv2d(logits, laplacian_kernel, padding=1, groups=channels)
        target_boundary = F.conv2d(targets, laplacian_kernel, padding=1, groups=channels)
        
        # 경계 가중치 맵 생성
        weight_map = F.max_pool2d(torch.sigmoid(target_boundary), 
                                 kernel_size=3, 
                                 stride=1, 
                                 padding=1)
        
        # BCE loss with logits 계산
        boundary_bce = F.binary_cross_entropy_with_logits(
            pred_boundary, 
            target_boundary,
            reduction='none'
        )
        
        # 최종 boundary loss 계산
        weighted_loss = boundary_bce * (1 + 2 * weight_map)
        return weighted_loss.mean()

    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets) * self.focal_weight
        dice = self.dice_loss(logits, targets) * self.dice_weight
        iou = self.iou_loss(logits, targets) * self.iou_weight
        ms_ssim = self.ms_ssim(logits, targets) * self.ms_ssim_weight
        boundary = self.boundary_loss(logits, targets) * self.boundary_weight
        
        # Combined loss
        # total_loss = focal + dice + iou + ms_ssim + boundary
        hybrid_loss = focal + ms_ssim + iou
        return hybrid_loss, focal, ms_ssim , iou
