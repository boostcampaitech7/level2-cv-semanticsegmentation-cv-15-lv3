import torch
import torch.nn as nn
import torchvision.models as models

from config import IMSIZE
import numpy as np
from Util.InitWeights import init_weights
from Util.SetSeed import set_seed
from .layer import unetConv2, BottleNeck
import torchvision.models as models
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision.models import resnet34, resnet50, resnet101,resnet152
import torch.nn.functional as F
set_seed()
from HRNet.lib.HRmodels.cls_hrnet import get_cls_net
import yaml


import torch
import torch.nn as nn
from HRNet.lib.HRmodels.cls_hrnet import HighResolutionNet, get_cls_net
import yaml


class HRNetEncoder_NOReduce(nn.Module): #1/4안줄이고 시작.
    def __init__(self, hrnet_config_file, pretrained_weights=None):
        super(HRNetEncoder_NOReduce, self).__init__()

        # Load HRNet configuration from YAML
        with open(hrnet_config_file, "r") as f:
            hrnet_config = yaml.safe_load(f)

        # Initialize HRNet
        self.hrnet = get_cls_net(hrnet_config)

        # Define new convolution layers
        self.init_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self.init_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        # Initialize weights for the new convolution layers
        self._initialize_new_conv_weights()

        # Load pretrained weights for HRNet
        if pretrained_weights:
            self.hrnet.init_weights(pretrained=pretrained_weights)
        
    def _initialize_new_conv_weights(self):
        """
        Initialize weights for the newly added convolution layers.
        """
        for module in [self.init_conv1, self.init_conv2]:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # Stage 1
        x = self.init_conv1(x)
        x = self.init_conv2(x)

        h1 = self.hrnet.layer1(x)  # First resolution (single scale)

        # Stage 2
        x_list = []
        for i in range(2):
            if self.hrnet.transition1[i] is not None:
                x_list.append(self.hrnet.transition1[i](h1))
            else:
                x_list.append(h1)
        y_list = self.hrnet.stage2(x_list)
        h2 = self._merge_multi_scale(y_list)  # Merge outputs for this stage

        # Stage 3
        x_list = []
        for i in range(3):
            if self.hrnet.transition2[i] is not None:
                x_list.append(self.hrnet.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.hrnet.stage3(x_list)
        h3 = self._merge_multi_scale(y_list)  # Merge outputs for this stage

        # Stage 4
        x_list = []
        for i in range(4):
            if self.hrnet.transition3[i] is not None:
                x_list.append(self.hrnet.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.hrnet.stage4(x_list)
        h4 = self._merge_multi_scale(y_list)  # Merge outputs for this stage
        #print(h1.shape,h2.shape,h3.shape,h4.shape)
        return h1, h2, h3, h4

    def _merge_multi_scale(self, features):
        """
        Merge multi-scale outputs into a single feature map by downsampling all to the lowest resolution.
        Args:
            features (list[torch.Tensor]): Multi-scale feature maps.
        Returns:
            torch.Tensor: Merged feature map.
        """
        # Determine the target size (lowest resolution in the list)
        target_size = features[-1].size()[2:]  # Height and Width of the last feature (lowest resolution)

        # Downsample all features to the target size and concatenate
        merged = torch.cat(
            [nn.functional.interpolate(feat, size=target_size, mode='bilinear', align_corners=True) if feat.size()[2:] != target_size else feat
             for feat in features],
            dim=1  # Concatenate along the channel dimension
        )
        return merged

class HRNetEncoder(nn.Module): #1/4이미지
    def __init__(self, hrnet_config_file, pretrained_weights=None):
        super(HRNetEncoder, self).__init__()

        # Load HRNet configuration from YAML
        with open(hrnet_config_file, "r") as f:
            hrnet_config = yaml.safe_load(f)

        # Initialize HRNet
        self.hrnet = get_cls_net(hrnet_config)

        # Load pretrained weights
        if pretrained_weights:
            self.hrnet.init_weights(pretrained_weights)
        
    def forward(self, x):
        # Stage 1
        x = self.hrnet.conv1(x)
        x = self.hrnet.bn1(x)
        x = self.hrnet.relu(x)
        x = self.hrnet.conv2(x)
        x = self.hrnet.bn2(x)
        x = self.hrnet.relu(x)
        h1 = self.hrnet.layer1(x)  # First resolution (single scale)
        #print("!!",h1.shape)
        # Stage 2
        x_list = []
        for i in range(2):
            if self.hrnet.transition1[i] is not None:
                x_list.append(self.hrnet.transition1[i](h1))
            else:
                x_list.append(h1)
        y_list = self.hrnet.stage2(x_list)
        #for a in y_list:
            #print("2:@@@@@@@@@@@@@@@",a.shape)
        h2 = self._merge_multi_scale(y_list)  # Merge outputs for this stage
        #print(h2.shape)

        # Stage 3
        x_list = []
        for i in range(3):
            if self.hrnet.transition2[i] is not None:
                x_list.append(self.hrnet.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.hrnet.stage3(x_list)
        #for a in y_list:
            #print("3:@@@@@@@@@@@@@@@",a.shape)
        h3 = self._merge_multi_scale(y_list)  # Merge outputs for this stage
        #print(h3.shape)
        # Stage 4
        x_list = []
        for i in range(4):
            if self.hrnet.transition3[i] is not None:
                x_list.append(self.hrnet.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.hrnet.stage4(x_list)
        #for a in y_list:
            #print("4:@@@@@@@@@@@@@@@",a.shape)
        h4 = self._merge_multi_scale(y_list)  # Merge outputs for this stage
        #print(h4.shape)
        return h1, h2, h3, h4


    def _merge_multi_scale(self, features):
        """
        Merge multi-scale outputs into a single feature map by downsampling all to the lowest resolution.
        Args:
            features (list[torch.Tensor]): Multi-scale feature maps.
        Returns:
            torch.Tensor: Merged feature map.
        """
        # Determine the target size (lowest resolution in the list)
        target_size = features[-1].size()[2:]  # Height and Width of the last feature (lowest resolution)

        # Downsample all features to the target size and concatenate
        merged = torch.cat(
            [nn.functional.interpolate(feat, size=target_size, mode='bilinear', align_corners=True) if feat.size()[2:] != target_size else feat
             for feat in features],
            dim=1  # Concatenate along the channel dimension
        )
        return merged



class UNet3PlusHRNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1,
                 hrnet_config_file="/data/ephemeral/home/MCG/level2-cv-semanticsegmentation-cv-15-lv3/UNet3+/Code/HRNet/experiments/w64.yaml",
                 pretrained_weights="/data/ephemeral/home/MCG/hrnetv2_w64_imagenet_pretrained.pth"):
        super(UNet3PlusHRNet, self).__init__()

        filters = [256, 192, 448, 960] #HRNetEncoder_NOReduce
        #filters = [256, 144, 336, 720] #HRNetEncoder
        # Define HRNet stages as encoder
        self.encoder=HRNetEncoder_NOReduce(hrnet_config_file=hrnet_config_file, pretrained_weights=pretrained_weights)


        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 4
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)


        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)


        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore5 = nn.Upsample(size=(IMSIZE, IMSIZE), mode='bilinear', align_corners=True)  # 512x512로 고정
        self.upscore4 = nn.Upsample(size=(IMSIZE, IMSIZE), mode='bilinear', align_corners=True)  # 512x512로 고정
        self.upscore3 = nn.Upsample(size=(IMSIZE, IMSIZE), mode='bilinear', align_corners=True)  # 512x512로 고정
        self.upscore2 = nn.Upsample(size=(IMSIZE, IMSIZE), mode='bilinear', align_corners=True)  # 512x512로 고정
        self.upscore1 = nn.Upsample(size=(IMSIZE, IMSIZE), mode='bilinear', align_corners=True)  # 512x512로 고정


        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.4),               # Dropout으로 오버피팅 방지
            nn.Conv2d(filters[3], n_classes, 1),  # 클래스 수 반영
            nn.AdaptiveMaxPool2d(1),         # 클래스별 전역 정보 추출
            nn.Sigmoid()                     # 멀티라벨 환경에서 클래스 존재 확률 출력
        )
        self.encoder_ids = {id(module) for module in self.encoder.modules()}

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights for the decoder layers only, excluding the encoder.
        """
        for module in self.modules():
            # Skip initialization if the module belongs to the encoder
            if id(module) in self.encoder_ids:
                continue
            if isinstance(module, nn.Conv2d):
                init_weights(module, init_type='kaiming')
            elif isinstance(module, nn.BatchNorm2d):
                init_weights(module, init_type='kaiming')
            elif isinstance(module, nn.ConvTranspose2d):
                init_weights(module, init_type='kaiming')
    
    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final
    
    def forward(self, inputs):
        ## -------------Encoder-------------
        #print(f"inputs shape: {inputs.shape}")
        h1,h2,h3,h4 = self.encoder(inputs)  # h1->320*320*64
        #print(h1.shape,h2.shape,h3.shape,h4.shape)

        # -------------Classification-------------
        cls_branch = self.cls(h4).squeeze(3).squeeze(2)  # (B, N, 1, 1) -> (B, N)

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        #print(h1_PT_hd4.shape,h2_PT_hd4.shape,h3_PT_hd4.shape,h4_Cat_hd4.shape)
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1), 1)))) # hd1->320*320*UpChannels
        #print(hd1.shape,hd2.shape,hd3.shape,hd4.shape)

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # 128->256

        d1 = self.outconv1(hd1) # 256
        d1 = self.upscore1(d1)
        
        d1 = self.dotProduct(d1, cls_branch)
        d2 = self.dotProduct(d2, cls_branch)
        d3 = self.dotProduct(d3, cls_branch)
        d4 = self.dotProduct(d4, cls_branch)
        #d5 = self.dotProduct(d5, cls_branch)
        
        # 가중치 적용
        weights = [0.42, 0.27, 0.17, 0.14]  # 가중치
        final_output = (
            weights[0] * d1 + 
            weights[1] * d2 + 
            weights[2] * d3 + 
            weights[3] * d4 
        )

        return final_output
        
        '''if self.training:
            return d1, d2, d3, d4, d5
        else:
            #print(d1)
            return d1'''