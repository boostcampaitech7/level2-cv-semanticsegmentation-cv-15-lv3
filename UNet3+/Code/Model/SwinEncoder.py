import timm
import torch.nn as nn

class SwinEncoder(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(SwinEncoder, self).__init__()
        self.swin = timm.create_model('swin_large_patch4_window12_384',
                                    pretrained=pretrained,
                                    features_only=True,
                                    out_indices=(0, 1, 2, 3),
                                    img_size=1024,
                                    in_chans=in_channels)
        
        # Swin-L의 출력을 UNet3+에 맞게 조정하는 1x1 컨볼루션
        self.adjust_channels = nn.ModuleList([
            nn.Conv2d(192, 64, 1),     # Stage 1: 192 -> 64
            nn.Conv2d(384, 256, 1),    # Stage 2: 384 -> 256
            nn.Conv2d(768, 512, 1),    # Stage 3: 768 -> 512
            nn.Conv2d(1536, 1024, 1),  # Stage 4: 1536 -> 1024
        ])
        
        # 마지막 feature map을 위한 추가 처리
        self.final_conv = nn.Sequential(
            nn.Conv2d(1536, 2048, 1),  # Stage 4: 1536 -> 2048
            nn.MaxPool2d(2, 2)         # 32x32 -> 16x16
        )
        
    def forward(self, x):
        # Swin Transformer features 추출
        features = self.swin(x)
        
        # 차원 순서 변경 (B, H, W, C) -> (B, C, H, W)
        features = [feat.permute(0, 3, 1, 2) for feat in features]
        
        # 채널 수 조정
        h1 = self.adjust_channels[0](features[0])  # 64 channels, 256x256
        h2 = self.adjust_channels[1](features[1])  # 256 channels, 128x128
        h3 = self.adjust_channels[2](features[2])  # 512 channels, 64x64
        h4 = self.adjust_channels[3](features[3])  # 1024 channels, 32x32
        h5 = self.final_conv(features[3])          # 2048 channels, 16x16
        
        return h1, h2, h3, h4, h5
