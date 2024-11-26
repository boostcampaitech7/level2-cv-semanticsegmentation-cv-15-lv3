import torch
import torch.nn as nn
from scipy.ndimage import sobel

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
        super(EdgeLoss, self).__init__()
        self.bce_loss = nn.BCELoss()  # BCE Loss for edge comparison

    def forward(self, pred, target):
        # Extract edges from target and predicted labels
        target_edges = self.get_edges(target)
        pred_edges = self.get_edges(torch.argmax(pred, dim=1))

        # Convert edges to float for loss calculation
        pred_edges = pred_edges.float()
        target_edges = target_edges.float()

        # Compute Edge Loss (e.g., BCE)
        edge_loss = self.bce_loss(pred_edges, target_edges)

        return edge_loss

    @staticmethod
    def get_edges(tensor):
        """
        Apply Sobel filter to extract edges.
        """
        tensor = tensor.detach().cpu().numpy()
        edges = sobel(tensor, axis=-1) + sobel(tensor, axis=-2)
        edges = torch.tensor((edges > 0).astype(float)).to(tensor.device)
        return edges

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iou_loss = LogIoULoss()
        self.alpha = alpha  # BCE와 IoU loss의 가중치 비율
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        iou = self.iou_loss(pred, target)
        return self.alpha * bce + (1 - self.alpha) * iou, bce, iou


