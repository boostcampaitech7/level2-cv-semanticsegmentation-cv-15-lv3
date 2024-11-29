import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from segmentation_models_pytorch.encoders._base import EncoderMixin
from config.config import Config
import timm

class SwinEncoder(torch.nn.Module, EncoderMixin):
    def __init__(self, **kwargs):
        super().__init__()
        
        # UnetPlusPlus에 맞는 채널 수로 수정
        self._out_channels = [3, 64, 128, 256, 512, 1024]  # 더 큰 채널 수 사용
        self._depth = 5
        self._in_channels = 3
        
        self.swin = timm.create_model(
            'swin_large_patch4_window12_384',
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=Config.IMG_SIZE,
            in_chans=self._in_channels
        )
        
        # 채널 수 조정
        self.adjust_channels = nn.ModuleList([
            nn.Conv2d(192, 64, 1),      # Stage 1: 192 -> 64
            nn.Conv2d(384, 128, 1),     # Stage 2: 384 -> 128
            nn.Conv2d(768, 256, 1),     # Stage 3: 768 -> 256
            nn.Conv2d(1536, 512, 1),    # Stage 4: 1536 -> 512
        ])
        
        # 마지막 feature map
        self.final_conv = nn.Sequential(
            nn.Conv2d(1536, 1024, 1),   # Stage 4: 1536 -> 1024
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.swin(x)
        features = [feat.permute(0, 3, 1, 2) for feat in features]
        
        outputs = []
        outputs.append(x)
        
        # 각 feature map을 원래 크기로 업샘플링
        for i, feat in enumerate(features):
            feat = self.adjust_channels[i](feat)
            feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
            outputs.append(feat)
        
        # 마지막 feature map도 업샘플링
        final_feat = self.final_conv(features[-1])
        final_feat = F.interpolate(final_feat, scale_factor=2, mode='bilinear', align_corners=False)
        outputs.append(final_feat)
        
        return outputs

    def load_state_dict(self, state_dict, **kwargs):
        self.swin.load_state_dict(state_dict, strict=False, **kwargs)