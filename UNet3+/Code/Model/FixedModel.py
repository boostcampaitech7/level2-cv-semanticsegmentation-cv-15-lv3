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

set_seed()

class UNet_3Plus_DeepSup(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True, pretrained=True):
        super(UNet_3Plus_DeepSup, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [192, 384, 768, 1536, 1536]

        # ConvNeXt Large Encoder
        self.convnext = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT).features
        #인코더 1 # H, W /filters[0]
        # 2 # H/2, W/2 / filters[1]
        # 3 # H/4, W/4 / filters[2]
        # 4 # H/8, W/8 / filters[3]
        # 5 # H/16, W/16 / filters[4]
        
        ## -------------Encoder--------------
        self.conv1 = nn.Sequential(
            unetConv2(self.in_channels, filters[0], self.is_batchnorm),
            nn.Dropout(p=0.05),
            BottleNeck(filters[0], filters[1]),
        )
        self.conv2 = self.convnext[1:3]

        # Replace conv3, conv4, and conv5 with ConvNeXt Stages
        self.conv3 = self.convnext[3:5]  # ConvNeXt Stage 2 (Output: 28x28, 384 channels)
        self.conv4 = self.convnext[5:7]
        self.conv5 = nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2),  # DownSample using MaxPool
        self.convnext[7:])


        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
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

        # hd5->20*20, hd4->40*40, Upsample 2 times (Using ConvTranspose2d)
        self.hd5_UT_hd4 = nn.ConvTranspose2d(
            in_channels=filters[4],       # 입력 채널
            out_channels=self.CatChannels,  # 출력 채널
            kernel_size=2,  # 업샘플링 크기
            stride=2        # 2배 업샘플링
        )
        self.hd5_UT_hd4_conv = nn.Conv2d(self.CatChannels, self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

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

        # hd4->40*40, hd4->80*80, Upsample 2 times (Using ConvTranspose2d)
        self.hd4_UT_hd3 = nn.ConvTranspose2d(
            in_channels=self.UpChannels,  # 입력 채널
            out_channels=self.CatChannels,  # 출력 채널
            kernel_size=2,  # 업샘플링 크기
            stride=2        # 2배 업샘플링
        )
        self.hd4_UT_hd3_conv = nn.Conv2d(self.CatChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times (Using ConvTranspose2d)
        self.hd5_UT_hd3 = nn.ConvTranspose2d(
            in_channels=filters[4],       # 입력 채널
            out_channels=self.CatChannels,  # 출력 채널
            kernel_size=4,  # 업샘플링 크기
            stride=4        # 4배 업샘플링
        )
        self.hd5_UT_hd3_conv = nn.Conv2d(self.CatChannels, self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

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

        # hd3->80*80, hd2->160*160, Upsample 2 times (Using ConvTranspose2d)
        self.hd3_UT_hd2 = nn.ConvTranspose2d(
            in_channels=self.UpChannels,  # 입력 채널
            out_channels=self.CatChannels,  # 출력 채널
            kernel_size=2,  # 업샘플링 크기
            stride=2        # 2배 업샘플링
        )
        self.hd3_UT_hd2_conv = nn.Conv2d(self.CatChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times (Using ConvTranspose2d)
        self.hd4_UT_hd2 = nn.ConvTranspose2d(
            in_channels=self.UpChannels,  # 입력 채널
            out_channels=self.CatChannels,  # 출력 채널
            kernel_size=4,  # 업샘플링 크기
            stride=4        # 4배 업샘플링
        )
        self.hd4_UT_hd2_conv = nn.Conv2d(self.CatChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times (Using ConvTranspose2d)
        self.hd5_UT_hd2 = nn.ConvTranspose2d(
            in_channels=filters[4],       # 입력 채널
            out_channels=self.CatChannels,  # 출력 채널
            kernel_size=8,  # 업샘플링 크기
            stride=8        # 8배 업샘플링
        )
        self.hd5_UT_hd2_conv = nn.Conv2d(self.CatChannels, self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)


        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times (Using ConvTranspose2d)
        self.hd2_UT_hd1 = nn.ConvTranspose2d(
            in_channels=self.UpChannels,  # 입력 채널
            out_channels=self.CatChannels,  # 출력 채널
            kernel_size=2,  # 업샘플링 크기
            stride=2        # 2배 업샘플링
        )
        self.hd2_UT_hd1_conv = nn.Conv2d(self.CatChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times (Using ConvTranspose2d)
        self.hd3_UT_hd1 = nn.ConvTranspose2d(
            in_channels=self.UpChannels,  # 입력 채널
            out_channels=self.CatChannels,  # 출력 채널
            kernel_size=4,  # 업샘플링 크기
            stride=4        # 4배 업샘플링
        )
        self.hd3_UT_hd1_conv = nn.Conv2d(self.CatChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times (Using ConvTranspose2d)
        self.hd4_UT_hd1 = nn.ConvTranspose2d(
            in_channels=self.UpChannels,  # 입력 채널
            out_channels=self.CatChannels,  # 출력 채널
            kernel_size=8,  # 업샘플링 크기
            stride=8        # 8배 업샘플링
        )
        self.hd4_UT_hd1_conv = nn.Conv2d(self.CatChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times (Using ConvTranspose2d)
        self.hd5_UT_hd1 = nn.ConvTranspose2d(
            in_channels=filters[4],       # 입력 채널
            out_channels=self.CatChannels,  # 출력 채널
            kernel_size=16,  # 업샘플링 크기
            stride=16        # 16배 업샘플링
        )
        self.hd5_UT_hd1_conv = nn.Conv2d(self.CatChannels, self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)


        # -------------Bilinear Upsampling--------------
        # -------------Learnable Upsampling using ConvTranspose2d--------------
        self.upscore6 = nn.ConvTranspose2d(
            in_channels=n_classes,   # 입력 채널 수
            out_channels=n_classes,  # 출력 채널 수 (Segmentation 결과 채널 유지)
            kernel_size=64,          # 업샘플링 커널 크기
            stride=32,               # 32배 업샘플링
            padding=16               # 출력 크기를 동일하게 맞추기 위한 패딩
        )

        self.upscore5 = nn.ConvTranspose2d(
            in_channels=n_classes,
            out_channels=n_classes,
            kernel_size=32,
            stride=16,
            padding=8
        )

        self.upscore4 = nn.ConvTranspose2d(
            in_channels=n_classes,
            out_channels=n_classes,
            kernel_size=16,
            stride=8,
            padding=4
        )

        self.upscore3 = nn.ConvTranspose2d(
            in_channels=n_classes,
            out_channels=n_classes,
            kernel_size=8,
            stride=4,
            padding=2
        )

        self.upscore2 = nn.ConvTranspose2d(
            in_channels=n_classes,
            out_channels=n_classes,
            kernel_size=4,
            stride=2,
            padding=1
        )

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, padding=1)
        
        self.cls = nn.Sequential(
            nn.Dropout(p=0.2),               # Dropout으로 오버피팅 방지
            nn.Conv2d(filters[4], n_classes, 1),  # 클래스 수 반영
            nn.AdaptiveMaxPool2d(1),         # 클래스별 전역 정보 추출
            nn.Sigmoid()                     # 멀티라벨 환경에서 클래스 존재 확률 출력
        )
        encoder_ids = {id(module) for module in self.convnext}  # ConvNeXt 모듈 ID 수집
        for module in self.modules():
            if id(module) in encoder_ids:
                #print(module)
                continue  # ConvNeXt 모듈 건너뜀
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
        h1 = self.conv1(inputs)  # h1->320*320*64
        #print(f"h1 shape: {h1.shape}")

        # h2 = self.maxpool1(h1)
        h2 = self.conv2(h1)  # h2->160*160*128
        #print(f"h2 shape: {h2.shape}")

        # h3 = self.maxpool2(h2)
        h3 = self.conv3(h2)  # h3->80*80*256
        #print(f"h3 shape: {h3.shape}")

        h4 = self.conv4(h3)
        #print(f"h4 shape: {h4.shape}")

        hd5 = self.conv5(h4)  # h5->20*20*1024
        #print(f"hd5 shape: {hd5.shape}")

        # -------------Classification-------------
        cls_branch = self.cls(hd5).squeeze(3).squeeze(2)  # (B, N, 1, 1) -> (B, N)
        threshold = 0.5
        cls_branch_mask = (cls_branch > threshold).float()  # (B, N), 각 클래스 존재 여부

        
        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))


        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5) # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # 128->256

        d1 = self.outconv1(hd1) # 256
        '''
        d1 = self.dotProduct(d1, cls_branch_mask)
        d2 = self.dotProduct(d2, cls_branch_mask)
        d3 = self.dotProduct(d3, cls_branch_mask)
        d4 = self.dotProduct(d4, cls_branch_mask)
        d5 = self.dotProduct(d5, cls_branch_mask)
        
        '''
        
        if self.training:
            return d1, d2, d3, d4, d5
        else:
            #print(d1)
            return d1