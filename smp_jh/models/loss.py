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
    def __init__(self, data_range=1.0, size_average=True, win_size=11, win_sigma=1.5, weights=None, scale_factor=0.5):
        super(MS_SSIM_Loss, self).__init__()
        self.ms_ssim = MS_SSIM(
            data_range=data_range,
            size_average=size_average,
            win_size=win_size,
            win_sigma=win_sigma,
            weights=weights,
            channel=1  # 채널별 처리를 위해 1로 설정
        )
        self.scale_factor = scale_factor
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.type_as(probs)
        
        # 이미지 크기 축소
        if self.scale_factor < 1.0:
            probs = F.interpolate(probs, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
            targets = F.interpolate(targets, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        total_loss = 0
        num_channels = probs.size(1)
        
        # 채널별로 순차 처리
        for c in range(num_channels):
            ms_ssim_val = self.ms_ssim(probs[:, c:c+1], targets[:, c:c+1])
            total_loss += (1 - ms_ssim_val)
        
        return total_loss / num_channels

# class HybridLoss(nn.Module):
#     def __init__(self, bce_weight=1, iou_weight=1, ms_ssim_weight=1, smooth=1e-6):
#         super(HybridLoss, self).__init__()
#         self.bce_weight = bce_weight
#         self.iou_weight = iou_weight
#         self.ms_ssim_weight = ms_ssim_weight
#         self.smooth = smooth
#         self.ms_ssim = MS_SSIM_Loss()
#         self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')  # BCE loss with logits

#     def adaptive_focal_loss(self, logits, targets, alpha=1, gamma_min=1.5, gamma_max=4.0, reduce=True):
#         # Compute BCE loss
#         BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')#self.bce_loss_fn(logits, targets)

#         # Compute pt (predicted probability for true class)
#         pt = torch.exp(-BCE_loss)

#         # Dynamically adjust gamma based on pt
#         gamma = gamma_min + (1 - pt) * (gamma_max - gamma_min)
#         gamma = torch.clamp(gamma, gamma_min, gamma_max)  # Ensure gamma stays within [gamma_min, gamma_max]

#         # Compute Focal Loss
#         F_loss = alpha * (1 - pt) ** gamma * BCE_loss

#         # Reduce loss if required
#         if reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss

#     def focal_loss(self, logits, targets, alpha=1, gamma=1.5, reduce=True):
#         BCE_loss= F.binary_cross_entropy_with_logits(logits, targets, reduction='none')#self.bce_loss_fn(logits, targets)
#         #print("BCE:",BCE_loss)
#         pt = torch.exp(-BCE_loss)
#         F_loss = alpha * (1-pt)**gamma * BCE_loss
#         if reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss.sum() / logits.size(0)

#     def iou_loss(self, logits, targets):
#         probs = torch.sigmoid(logits)
#         intersection = (probs * targets).sum(dim=(2, 3))
#         union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
#         iou_loss = -torch.log((intersection + self.smooth) / (union + self.smooth))
#         return iou_loss.mean()

#     def forward(self, logits, targets):
#         bce = self.bce_loss_fn(logits, targets) * self.bce_weight
#         iou = self.iou_loss(logits, targets) * self.iou_weight
#         ms_ssim = self.ms_ssim(logits, targets) * self.ms_ssim_weight
        
#         hybrid_loss = bce + iou
#         return hybrid_loss, bce ,iou

# class HybridLoss(nn.Module):
#     def __init__(self, bce_weight=0.4, iou_weight=0.4, ms_ssim_weight=0.2, smooth=1e-6):
#         super(HybridLoss, self).__init__()
#         self.bce_weight = bce_weight
#         self.iou_weight = iou_weight
#         self.ms_ssim_weight = ms_ssim_weight
#         self.smooth = smooth
#         self.ms_ssim = MS_SSIM_Loss()
#         self.pos_weight = None  # 초기화는 forward에서 처리
#         self.bce_loss_fn = None  # 초기화는 forward에서 처리
    
#     def iou_loss(self, logits, targets):
#         probs = torch.sigmoid(logits)
        
#         # batch size별 처리
#         batch_size = probs.size(0)
#         total_loss = 0
        
#         for i in range(batch_size):
#             intersection = (probs[i] * targets[i]).sum()
#             union = probs[i].sum() + targets[i].sum() - intersection
            
#             # 클래스별 가중치 적용 (target에 있는 클래스에 더 높은 가중치)
#             weight = 2.0 if targets[i].sum() > 0 else 1.0
            
#             iou = (intersection + self.smooth) / (union + self.smooth)
#             total_loss += weight * (1 - iou)
            
#         return total_loss / batch_size
        
#     def forward(self, logits, targets):
#         # BCE Loss 초기화 (첫 forward pass에서)
#         if self.bce_loss_fn is None:
#             self.pos_weight = torch.tensor([2.0]).to(logits.device)
#             self.bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            
#         # Loss 계산
#         bce = self.bce_loss_fn(logits, targets)
#         iou = self.iou_loss(logits, targets)
#         ms_ssim = self.ms_ssim(logits, targets)
        
#         # Loss 스케일 정규화
#         bce = torch.clamp(bce, 0, 1.0) * self.bce_weight
#         iou = torch.clamp(iou, 0, 1.0) * self.iou_weight
#         ms_ssim = torch.clamp(ms_ssim, 0, 1.0) * self.ms_ssim_weight
        
#         # Loss 유효성 검사
#         if torch.isnan(bce) or torch.isinf(bce):
#             bce = torch.tensor(0.0, device=logits.device)
#         if torch.isnan(iou) or torch.isinf(iou):
#             iou = torch.tensor(0.0, device=logits.device)
#         if torch.isnan(ms_ssim) or torch.isinf(ms_ssim):
#             ms_ssim = torch.tensor(0.0, device=logits.device)
        
#         hybrid_loss = bce + iou + ms_ssim
        
#         return hybrid_loss, bce, ms_ssim, iou

class HybridLoss(nn.Module):
    def __init__(self, bce_weight=1.0, iou_weight=1.0, anatomical_weight=0.5, smooth=1e-6):
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.anatomical_weight = anatomical_weight
        self.smooth = smooth
        
        # 손목뼈 클래스 인덱스 정의
        self.wrist_pairs = [
            ('Trapezium', 'Trapezoid'),
            ('Triquetrum', 'Pisiform'),
            ('Scaphoid', 'Lunate'),
            ('Lunate', 'Triquetrum'),
            ('Capitate', 'Hamate'),
            ('Lunate', 'Capitate'),
            ('Scaphoid', 'Capitate')
        ]
        self.wrist_indices = [(Config.CLASS2IND[b1], Config.CLASS2IND[b2]) 
                             for b1, b2 in self.wrist_pairs]
        
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    def anatomical_consistency_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        total_loss = torch.tensor(0.0, device=logits.device)  # 명시적으로 device 지정
        
        for idx1, idx2 in self.wrist_indices:
            # 두 뼈의 예측값 (B, H, W) 형태로 가져오기
            pred1, pred2 = probs[:, idx1], probs[:, idx2]
            target1, target2 = targets[:, idx1], targets[:, idx2]
            
            # 1. 겹침 영역에서의 일관성 loss
            overlap_region = (pred1 > 0.5) & (pred2 > 0.5)
            if overlap_region.any():
                overlap_loss = torch.abs(pred1[overlap_region] - pred2[overlap_region]).mean()
                total_loss = total_loss + overlap_loss
            
            # 2. 거리 기반 제약
            if (target1.sum() > 0) and (target2.sum() > 0):
                # 차원 추가 (B, 1, H, W)
                pred1_4d = pred1.unsqueeze(1)
                pred2_4d = pred2.unsqueeze(1)
                target1_4d = target1.unsqueeze(1)
                target2_4d = target2.unsqueeze(1)
                
                # 중심점 계산
                center1 = self.get_center_of_mass(pred1_4d)  # (B, 2)
                center2 = self.get_center_of_mass(pred2_4d)  # (B, 2)
                target_center1 = self.get_center_of_mass(target1_4d)  # (B, 2)
                target_center2 = self.get_center_of_mass(target2_4d)  # (B, 2)
                
                # 거리 계산 및 loss 추가
                pred_dist = torch.norm(center1 - center2, p=2, dim=1)  # (B,)
                target_dist = torch.norm(target_center1 - target_center2, p=2, dim=1)  # (B,)
                dist_loss = torch.abs(pred_dist - target_dist).mean()
                
                total_loss = total_loss + 0.1 * dist_loss
        
        return total_loss / len(self.wrist_indices)

    def get_center_of_mass(self, mask):
        # mask shape: (B, 1, H, W)
        B, _, H, W = mask.shape
        
        # 그리드 생성
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=mask.device).float(),
                                    torch.arange(W, device=mask.device).float(),
                                    indexing='ij')
        
        # 배치 차원 추가
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        
        # 중심점 계산
        total_mass = mask.sum(dim=(2,3)) + 1e-6  # (B, 1)
        center_y = (mask * grid_y).sum(dim=(2,3)) / total_mass  # (B, 1)
        center_x = (mask * grid_x).sum(dim=(2,3)) / total_mass  # (B, 1)
        
        return torch.cat([center_x, center_y], dim=1)  # (B, 2)

    def get_center_distance(self, mask1, mask2):
        center1 = self.get_center_of_mass(mask1)  # (B, 2, 1, 1)
        center2 = self.get_center_of_mass(mask2)  # (B, 2, 1, 1)
        return torch.norm(center1 - center2, p=2, dim=1)  # (B, 1, 1)
    
    def iou_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
        iou_loss = -torch.log((intersection + self.smooth) / (union + self.smooth))
        return iou_loss.mean()

    def forward(self, logits, targets):
        # 기본 loss 계산
        bce = self.bce_loss_fn(logits, targets).mean() * self.bce_weight
        iou = self.iou_loss(logits, targets) * self.iou_weight
        
        # 해부학적 일관성 loss 추가
        anatomical = self.anatomical_consistency_loss(logits, targets) * self.anatomical_weight
        
        hybrid_loss = bce + iou + anatomical
        return hybrid_loss, bce, iou, anatomical