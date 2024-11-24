import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import IMSIZE

class SeparatedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, size=3, padding='same'):
        super(SeparatedConv2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, size), padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (size, 1), padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class MidScopeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MidScopeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class WideScopeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WideScopeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=3, dilation=3)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(ResNetBlock, self).__init__()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=dilation_rate)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation_rate, dilation=dilation_rate)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=dilation_rate, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + shortcut
        x = self.relu(x)
        return x

class DuckV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, size=3, memory_efficient=True):
        super(DuckV2Block, self).__init__()
        self.memory_efficient = memory_efficient
        self.bn_in = nn.BatchNorm2d(in_channels)
        
        mid_channels = out_channels // 2 if memory_efficient else out_channels
        
        self.wide_scope = WideScopeBlock(in_channels, mid_channels)
        self.mid_scope = MidScopeBlock(in_channels, mid_channels)
        
        self.resnet1 = ResNetBlock(in_channels, mid_channels)
        self.resnet2 = ResNetBlock(in_channels, mid_channels)
        
        self.separated = SeparatedConv2D(in_channels, mid_channels, size=6)
        
        self.final_conv = nn.Conv2d(mid_channels * 4, out_channels, 1)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn_in(x)
        
        outputs = []
        
        x1 = self.wide_scope(x)
        outputs.append(x1)
        
        x2 = self.mid_scope(x)
        outputs.append(x2)
        
        x3 = self.resnet1(x)
        outputs.append(x3)
        
        x4 = self.resnet2(x)
        outputs.append(x4)
        
        x = torch.cat(outputs, dim=1)
        del outputs
        
        x = self.final_conv(x)
        x = self.bn_out(x)
        
        return x

class SwinEncoder(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(SwinEncoder, self).__init__()
        self.swin = timm.create_model('swin_large_patch4_window12_384',
                                    pretrained=pretrained,
                                    features_only=True,
                                    out_indices=(0, 1, 2, 3),
                                    img_size=IMSIZE,
                                    in_chans=in_channels)
        
        # 각 stage별로 적확한 크기와 채널 수로 조정
        self.conv1 = nn.Sequential(
            DuckV2Block(192, 64),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )  # -> H
        
        self.conv2 = nn.Sequential(
            DuckV2Block(384, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )  # -> H/2
        
        self.conv3 = DuckV2Block(768, 256)  # -> H/4
        
        self.conv4 = nn.Sequential(
            DuckV2Block(1536, 512),
            nn.MaxPool2d(2, 2)
        )  # -> H/8
        
        self.conv5 = nn.Sequential(
            DuckV2Block(1536, 1024),  # 원래 채널 수로 복원
            nn.MaxPool2d(4, 4)
        )  # -> H/16

    def forward(self, x):
        features = self.swin(x)
        features = [feat.permute(0, 3, 1, 2) for feat in features]
        
        h1 = self.conv1(features[0])
        h2 = self.conv2(features[1])
        h3 = self.conv3(features[2])
        h4 = self.conv4(features[3])
        h5 = self.conv5(features[3])
        
        return h1, h2, h3, h4, h5

class Duck3P(nn.Module):
    def __init__(self, in_channels=3, n_classes=1):
        super(Duck3P, self).__init__()
        
        self.filters = [64, 128, 256, 512, 1024]  # 채널 수 유지
        self.CatChannels = self.filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        # Encoder
        self.encoder = SwinEncoder(in_channels=in_channels)
        
        # Stage 4
        self.hd5_UT_hd4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DuckV2Block(self.filters[4], self.CatChannels)  # 1024 -> 64
        )
        
        # stage 4d
        self.h1_PT_hd4 = nn.Sequential(
            nn.MaxPool2d(8, 8, ceil_mode=True),
            DuckV2Block(self.filters[0], self.CatChannels)
        )
        
        self.h2_PT_hd4 = nn.Sequential(
            nn.MaxPool2d(4, 4, ceil_mode=True),
            DuckV2Block(self.filters[1], self.CatChannels)
        )
        
        self.h3_PT_hd4 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            DuckV2Block(self.filters[2], self.CatChannels)
        )
        
        self.h4_Cat_hd4 = DuckV2Block(self.filters[3], self.CatChannels)
        
        self.hd5_UT_hd4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DuckV2Block(self.filters[4], self.CatChannels)
        )
        
        self.conv4d_1 = DuckV2Block(self.UpChannels, self.UpChannels)

        # stage 3d
        self.h1_PT_hd3 = nn.Sequential(
            nn.MaxPool2d(4, 4, ceil_mode=True),
            DuckV2Block(self.filters[0], self.CatChannels)
        )
        
        self.h2_PT_hd3 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            DuckV2Block(self.filters[1], self.CatChannels)
        )
        
        self.h3_Cat_hd3 = DuckV2Block(self.filters[2], self.CatChannels)
        
        self.hd4_UT_hd3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DuckV2Block(self.UpChannels, self.CatChannels)
        )
        
        self.hd5_UT_hd3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            DuckV2Block(self.filters[4], self.CatChannels)
        )
        
        self.conv3d_1 = DuckV2Block(self.UpChannels, self.UpChannels)

        # stage 2d
        self.h1_PT_hd2 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            DuckV2Block(self.filters[0], self.CatChannels)
        )
        
        self.h2_Cat_hd2 = DuckV2Block(self.filters[1], self.CatChannels)
        
        self.hd3_UT_hd2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DuckV2Block(self.UpChannels, self.CatChannels)
        )
        
        self.hd4_UT_hd2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            DuckV2Block(self.UpChannels, self.CatChannels)
        )
        
        self.hd5_UT_hd2 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            DuckV2Block(self.filters[4], self.CatChannels)
        )
        
        self.conv2d_1 = DuckV2Block(self.UpChannels, self.UpChannels)

        # stage 1d
        self.h1_Cat_hd1 = DuckV2Block(self.filters[0], self.CatChannels)
        
        self.hd2_UT_hd1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DuckV2Block(self.UpChannels, self.CatChannels)
        )
        
        self.hd3_UT_hd1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            DuckV2Block(self.UpChannels, self.CatChannels)
        )
        
        self.hd4_UT_hd1 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            DuckV2Block(self.UpChannels, self.CatChannels)
        )
        
        self.hd5_UT_hd1 = nn.Sequential(
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
            DuckV2Block(self.filters[4], self.CatChannels)
        )
        
        self.conv1d_1 = DuckV2Block(self.UpChannels, self.UpChannels)

        # Output layers
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(self.filters[4], n_classes, 3, padding=1)

        # Final upsampling
        self.upscore = nn.Upsample(size=(IMSIZE, IMSIZE), mode='bilinear', align_corners=True)

    def forward(self, x):
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, 
                x,
                use_reentrant=False,
                preserve_rng_state=False
            )
        return self._forward(x)

    def _forward(self, x):
        # Encoder
        h1, h2, h3, h4, h5 = self.encoder(x)
        
        # Stage 4
        h1_PT_hd4 = self.h1_PT_hd4(h1)
        h2_PT_hd4 = self.h2_PT_hd4(h2)
        h3_PT_hd4 = self.h3_PT_hd4(h3)
        h4_Cat_hd4 = self.h4_Cat_hd4(h4)
        hd5_UT_hd4 = self.hd5_UT_hd4(h5)
        
        # Stage 4 크기 맞추기
        target_size_4 = h4_Cat_hd4.shape[2:]
        h1_PT_hd4 = F.interpolate(h1_PT_hd4, size=target_size_4, mode='bilinear', align_corners=True)
        h2_PT_hd4 = F.interpolate(h2_PT_hd4, size=target_size_4, mode='bilinear', align_corners=True)
        h3_PT_hd4 = F.interpolate(h3_PT_hd4, size=target_size_4, mode='bilinear', align_corners=True)
        hd5_UT_hd4 = F.interpolate(hd5_UT_hd4, size=target_size_4, mode='bilinear', align_corners=True)
        
        conv4d_1 = self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))
        
        # Stage 3
        h1_PT_hd3 = self.h1_PT_hd3(h1)
        h2_PT_hd3 = self.h2_PT_hd3(h2)
        h3_Cat_hd3 = self.h3_Cat_hd3(h3)
        hd4_UT_hd3 = self.hd4_UT_hd3(conv4d_1)
        hd5_UT_hd3 = self.hd5_UT_hd3(h5)
        
        # Stage 3 크기 맞추기
        target_size_3 = h3_Cat_hd3.shape[2:]
        h1_PT_hd3 = F.interpolate(h1_PT_hd3, size=target_size_3, mode='bilinear', align_corners=True)
        h2_PT_hd3 = F.interpolate(h2_PT_hd3, size=target_size_3, mode='bilinear', align_corners=True)
        hd4_UT_hd3 = F.interpolate(hd4_UT_hd3, size=target_size_3, mode='bilinear', align_corners=True)
        hd5_UT_hd3 = F.interpolate(hd5_UT_hd3, size=target_size_3, mode='bilinear', align_corners=True)
        
        conv3d_1 = self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))
        
        # Stage 2
        h1_PT_hd2 = self.h1_PT_hd2(h1)
        h2_Cat_hd2 = self.h2_Cat_hd2(h2)
        hd3_UT_hd2 = self.hd3_UT_hd2(conv3d_1)
        hd4_UT_hd2 = self.hd4_UT_hd2(conv4d_1)
        hd5_UT_hd2 = self.hd5_UT_hd2(h5)
        
        # Stage 2 크기 맞추기
        target_size_2 = h2_Cat_hd2.shape[2:]
        h1_PT_hd2 = F.interpolate(h1_PT_hd2, size=target_size_2, mode='bilinear', align_corners=True)
        hd3_UT_hd2 = F.interpolate(hd3_UT_hd2, size=target_size_2, mode='bilinear', align_corners=True)
        hd4_UT_hd2 = F.interpolate(hd4_UT_hd2, size=target_size_2, mode='bilinear', align_corners=True)
        hd5_UT_hd2 = F.interpolate(hd5_UT_hd2, size=target_size_2, mode='bilinear', align_corners=True)
        
        conv2d_1 = self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))
        
        # Stage 1
        h1_Cat_hd1 = self.h1_Cat_hd1(h1)
        hd2_UT_hd1 = self.hd2_UT_hd1(conv2d_1)
        hd3_UT_hd1 = self.hd3_UT_hd1(conv3d_1)
        hd4_UT_hd1 = self.hd4_UT_hd1(conv4d_1)
        hd5_UT_hd1 = self.hd5_UT_hd1(h5)
        
        # Stage 1 크기 맞추기
        target_size_1 = h1_Cat_hd1.shape[2:]
        hd2_UT_hd1 = F.interpolate(hd2_UT_hd1, size=target_size_1, mode='bilinear', align_corners=True)
        hd3_UT_hd1 = F.interpolate(hd3_UT_hd1, size=target_size_1, mode='bilinear', align_corners=True)
        hd4_UT_hd1 = F.interpolate(hd4_UT_hd1, size=target_size_1, mode='bilinear', align_corners=True)
        hd5_UT_hd1 = F.interpolate(hd5_UT_hd1, size=target_size_1, mode='bilinear', align_corners=True)
        
        conv1d_1 = self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))
        
        # Deep Supervision
        d5 = self.upscore(self.outconv5(h5))
        d4 = self.upscore(self.outconv4(conv4d_1))
        d3 = self.upscore(self.outconv3(conv3d_1))
        d2 = self.upscore(self.outconv2(conv2d_1))
        d1 = self.upscore(self.outconv1(conv1d_1))

        if self.training:
            return [d1, d2, d3, d4, d5]
        else:
            return d1
