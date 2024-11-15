# unetplusplus.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# from CustomLayers.ConvBlock2D import conv_block_2D

class ConvBlockNested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(ConvBlockNested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.activation(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, n1=64, height=512, width=512, supervision=True):
        super(UNetPlusPlus, self).__init__()

        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.ModuleList([
            nn.Upsample(size=(height // (2 ** c), width // (2 ** c)), mode='bilinear', align_corners=True) 
            for c in range(4)
        ])
        self.supervision = supervision

        self.conv0_0 = ConvBlockNested(in_ch, filters[0], filters[0])
        self.conv1_0 = ConvBlockNested(filters[0], filters[1], filters[1])
        self.conv2_0 = ConvBlockNested(filters[1], filters[2], filters[2])
        self.conv3_0 = ConvBlockNested(filters[2], filters[3], filters[3])
        self.conv4_0 = ConvBlockNested(filters[3], filters[4], filters[4])

        self.conv0_1 = ConvBlockNested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = ConvBlockNested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = ConvBlockNested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = ConvBlockNested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = ConvBlockNested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = ConvBlockNested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = ConvBlockNested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = ConvBlockNested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = ConvBlockNested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = ConvBlockNested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.seg_outputs = nn.ModuleList([
            nn.Conv2d(filters[0], out_ch, kernel_size=1, padding=0) for _ in range(4)
        ])

    def forward(self, x):
        seg_outputs = []

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up[0](x1_0)], 1))
        seg_outputs.append(self.seg_outputs[0](x0_1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up[1](x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up[0](x1_1)], 1))
        seg_outputs.append(self.seg_outputs[1](x0_2))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up[2](x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up[1](x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up[0](x1_2)], 1))
        seg_outputs.append(self.seg_outputs[2](x0_3))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up[3](x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up[2](x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up[1](x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up[0](x1_3)], 1))
        seg_outputs.append(self.seg_outputs[3](x0_4))

        if self.supervision:
            return seg_outputs
        else:
            return seg_outputs[-1]

class DUCKNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, height=512, width=512, starting_filters=64, supervision=False):
        super(DUCKNet, self).__init__()
        
        self.kernel_initializer = 'he_uniform'
        self.interpolation = "bilinear"
        
        # Down-sampling layers
        self.p1 = nn.Conv2d(in_ch, starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.p2 = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.p3 = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.p4 = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.p5 = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=0)
        
        # Initial conv block
        self.t0 = ConvBlock2D(in_ch, starting_filters, 'duckv2', repeat=1)
        
        # Intermediate layers with additions
        self.l1i = nn.Conv2d(starting_filters, starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.l2i = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.l3i = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.l4i = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.l5i = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=0)
        
        # conv_block layers
        self.t1 = ConvBlock2D(starting_filters * 2, starting_filters * 2, 'duckv2', repeat=1)
        self.t2 = ConvBlock2D(starting_filters * 4, starting_filters * 4, 'duckv2', repeat=1)
        self.t3 = ConvBlock2D(starting_filters * 8, starting_filters * 8, 'duckv2', repeat=1)
        self.t4 = ConvBlock2D(starting_filters * 16, starting_filters * 16, 'duckv2', repeat=1)
        
        # Deeper layers
        self.t51 = ConvBlock2D(starting_filters * 32, starting_filters * 32, 'resnet', repeat=2)
        self.t53 = ConvBlock2D(starting_filters * 32, starting_filters * 16, 'resnet', repeat=2)
        
        # Up-sampling layers를 동적으로 설정
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=self.interpolation, align_corners=True),
            nn.Conv2d(starting_filters * 16, starting_filters * 8, kernel_size=1)  # 채널 수 조정
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=self.interpolation, align_corners=True),
            nn.Conv2d(starting_filters * 8, starting_filters * 4, kernel_size=1)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=self.interpolation, align_corners=True),
            nn.Conv2d(starting_filters * 4, starting_filters * 2, kernel_size=1)
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=self.interpolation, align_corners=True),
            nn.Conv2d(starting_filters * 2, starting_filters, kernel_size=1)
        )
        
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=self.interpolation, align_corners=True),
            nn.Conv2d(starting_filters, starting_filters, kernel_size=1)
        )

        # Final upsampling and output
        self.z1 = ConvBlock2D(starting_filters, starting_filters, 'duckv2', repeat=1)
        
        self.output = nn.Conv2d(starting_filters, out_ch, kernel_size=1, stride=1)

    def forward(self, x):
        # Encoder pathway
        p1 = self.p1(x)          # /2
        p2 = self.p2(p1)         # /4
        p3 = self.p3(p2)         # /8
        p4 = self.p4(p3)         # /16
        p5 = self.p5(p4)         # /32
        
        t0 = self.t0(x)          # 원본 크기
        
        s1 = self.l1i(t0) + p1   # /2
        t1 = self.t1(s1)
        
        s2 = self.l2i(t1) + p2   # /4
        t2 = self.t2(s2)
        
        s3 = self.l3i(t2) + p3   # /8
        t3 = self.t3(s3)
        
        s4 = self.l4i(t3) + p4   # /16
        t4 = self.t4(s4)
        
        s5 = self.l5i(t4) + p5   # /32
        t51 = self.t51(s5)
        t53 = self.t53(t51)      # /32 -> /16
        
        # Decoder pathway with size matching
        l5o = self.up4(t53)      # /16 -> /8
        c4 = l5o + t3            # 이제 크기가 같음
        q4 = self.q4(c4)
        
        l4o = self.up3(q4)       # /8 -> /4
        c3 = l4o + t2
        q3 = self.q3(c3)
        
        l3o = self.up2(q3)       # /4 -> /2
        c2 = l3o + t1
        q6 = self.q6(c2)
        
        l2o = self.up1(q6)       # /2 -> 원본
        c1 = l2o + t0
        q1 = self.q1(c1)
        
        l1o = self.up0(q1)
        c0 = l1o + t0
        z1 = self.z1(c0)
        
        output = self.output(z1)
        
        if self.supervision:
            return [output, output, output, output]
        else:
            return output

class DuckV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DuckV2Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, block_type, repeat=1):
        super(ConvBlock2D, self).__init__()
        self.block_type = block_type
        
        if block_type == 'duckv2':
            self.blocks = nn.ModuleList([
                DuckV2Block(
                    in_channels if i == 0 else out_channels,
                    out_channels
                ) for i in range(repeat)
            ])
        elif block_type == 'resnet':
            # ResNet 블록 구현
            self.blocks = nn.ModuleList([
                ResNetBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels
                ) for i in range(repeat)
            ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# ResNet 블록 추가
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection을 위한 1x1 conv (채널 수가 다른 경우)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

# 반복적으로 나오는 구조를 쉽게 만들기 위해서 정의한 유틸리티 함수 입니다
def conv_relu(in_ch, out_ch, size=3, rate=1):
    conv_relu = nn.Sequential(
        nn.Conv2d(in_ch,
                  out_ch,
                  kernel_size=size,
                  stride=1,
                  padding=rate,
                  dilation=rate),
        nn.ReLU()
    )
    return conv_relu


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # TODO: DilatedNet 모델의 backbone으로 사용할 VGG16 모델을 완성하기 위해
        #    필요한 모듈들을 작성하세요
        self.features1 = nn.Sequential(
            conv_relu(3, 64, 3, 1),
            conv_relu(64, 64, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), 
        )
        self.features2 = nn.Sequential(
            conv_relu(64, 128, 3, 1),
            conv_relu(128, 128, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), 
        )

        self.features3 = nn.Sequential(
            conv_relu(128, 256, 3, 1),
            conv_relu(256, 256, 3, 1),
            conv_relu(256, 256, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), 
        )
        
        self.features4 = nn.Sequential(
            conv_relu(256, 512, 3, 1),
            conv_relu(512, 512, 3, 1),
            conv_relu(512, 512, 3, 1),
        )
        
        self.features5 = nn.Sequential(
            conv_relu(512, 512, 3, 2),
            conv_relu(512, 512, 3, 2),
            conv_relu(512, 512, 3, 2),
        )
        
    def forward(self, x):
        # TODO: `__init__`에서 작성한 모듈을 이용하여 forward 함수를 작성하세요
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        # TODO: backbone에서 얻은 피처의 차원을 클래스의 개수와 같게 만들어
        #    classification을 수행할 수 있도록 만드는 레이어를 작성합니다
        # cnn 구조를 활용한 classifier 작성
        # 층은 3개로 구성되어 있음, 처음에 커널 사이즈 7, 나머지 커널 사이즈 1
        # 활성 함수로는 relu 사용, dropout 0.5
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, dilation=4, padding=12),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(4096, num_classes, kernel_size=1),
        )
    def forward(self, x):
        # TODO: `__init__`에서 작성한 모듈을 이용하여 forward 함수를 작성하세요
        x = self.classifier(x)
        return x


class BasicContextModule(nn.Module):
    def __init__(self, num_classes):
        super(BasicContextModule, self).__init__()

        # TODO: BasicContextModule을 구성하는 모듈들을 작성합니다.
        #     BasicContextModule은 다양한 크기의 dilation rate를 가지는 dilated convolution을 순차적으로 적용합니다
        self.layer1 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 1))            
        self.layer2 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 1))            
        self.layer3 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 2))            
        self.layer4 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 4))            
        self.layer5 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 8))            
        self.layer6 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 16))            
        self.layer7 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 1))            
        self.layer8 = nn.Sequential(nn.Conv2d(num_classes, num_classes, 1, 1))            
        
        
    def forward(self, x):

        # TODO: `__init__`에서 작성한 모듈을 이용여 forward 함수를 작성하세요
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x


class DilatedNet(nn.Module):
    def __init__(self, backbone, classifier, context_module):
        super(DilatedNet, self).__init__()
        # TODO: DilatedNet 모델을 완성하기 위해 필요한 모듈들을 작성하세요
        #   상단에서 작성한 backbone VGG16 모델과 classifier 및 basic context module을 사용합니다
        self.backbone = backbone
        self.classifier = classifier
        self.context_module = context_module
        self.deconv = nn.ConvTranspose2d(in_channels=29, out_channels=29, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        # TODO: `__init__`에서 작성한 모듈을 이용하여 forward 함수를 작성하세요
        x = self.backbone(x)
        x = self.classifier(x)
        x = self.context_module(x)
        x = self.deconv(x)
        return x
