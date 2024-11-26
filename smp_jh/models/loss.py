import torch
import torch.nn as nn
import torch.nn.functional as F

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

class HybridLoss(nn.Module):
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


