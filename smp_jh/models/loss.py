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
        super().__init__()
        # Focal Loss 구현

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

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        
    def get_edges(self, tensor):
        # tensor를 numpy로 변환하지 않고 PyTorch operations 사용
        batch_size = tensor.size(0)
        
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
        
        # Apply convolution for edge detection
        tensor = tensor.view(-1, 1, tensor.size(2), tensor.size(3))  # reshape for conv2d
        edges_x = F.conv2d(tensor, sobel_x, padding=1)
        edges_y = F.conv2d(tensor, sobel_y, padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        
        # Threshold
        edges = (edges > 0.5).float()
        return edges.view(batch_size, -1, edges.size(2), edges.size(3))
    
    def forward(self, pred, target):
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        return self.criterion(pred_edges, target_edges)

class BIELoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iou_loss = LogIoULoss()
        self.edge_loss = EdgeLoss()
        self.alpha = alpha  # BCE weight
        self.beta = beta   # IoU weight
        self.gamma = gamma # Edge weight
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        iou = self.iou_loss(pred, target)
        edge = self.edge_loss(pred, target)
        
        total_loss = (self.alpha * bce + 
                     self.beta * iou + 
                     self.gamma * edge)
        
        return total_loss, bce, iou, edge

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
    def __init__(self, focal_weight=1, iou_weight=1, ms_ssim_weight=1, smooth=1e-6):
        super(HybridLoss, self).__init__()
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight
        self.ms_ssim_weight = ms_ssim_weight
        self.smooth = smooth
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

    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets) * self.focal_weight
        iou = self.iou_loss(logits, targets) * self.iou_weight
        ms_ssim = self.ms_ssim(logits, targets) * self.ms_ssim_weight
        
        hybrid_loss = focal + ms_ssim + iou
        return hybrid_loss, focal, ms_ssim , iou



