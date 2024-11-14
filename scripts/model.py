# unetplusplus.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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

import torch
import torch.nn as nn
import torch.nn.functional as F
# from CustomLayers.ConvBlock2D import conv_block_2D

class DUCKNet(nn.Module):
    def __init__(self, img_height, img_width, input_channels, out_classes, starting_filters):
        super(DUCKNet, self).__init__()
        
        self.kernel_initializer = 'he_uniform'
        self.interpolation = "nearest"
        
        # Down-sampling layers
        self.p1 = nn.Conv2d(input_channels, starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.p2 = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.p3 = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.p4 = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.p5 = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=0)
        
        # Initial conv block
        self.t0 = ConvBlock2D(input_channels, starting_filters, 'duckv2', repeat=1)
        
        # Intermediate layers with additions
        self.l1i = nn.Conv2d(starting_filters, starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.l2i = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.l3i = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.l4i = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=0)
        self.l5i = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=0)
        
        # conv_block layers with concatenation
        self.t1 = ConvBlock2D(starting_filters * 2, starting_filters * 2, 'duckv2', repeat=1)
        self.t2 = ConvBlock2D(starting_filters * 4, starting_filters * 4, 'duckv2', repeat=1)
        self.t3 = ConvBlock2D(starting_filters * 8, starting_filters * 8, 'duckv2', repeat=1)
        self.t4 = ConvBlock2D(starting_filters * 16, starting_filters * 16, 'duckv2', repeat=1)
        
        # Deeper layers
        self.t51 = ConvBlock2D(starting_filters * 32, starting_filters * 32, 'resnet', repeat=2)
        self.t53 = ConvBlock2D(starting_filters * 32, starting_filters * 16, 'resnet', repeat=2)
        
        # Up-sampling layers
        self.up4 = nn.Upsample(scale_factor=2, mode=self.interpolation)
        self.q4 = ConvBlock2D(starting_filters * 8, starting_filters * 8, 'duckv2', repeat=1)

        self.up3 = nn.Upsample(scale_factor=2, mode=self.interpolation)
        self.q3 = ConvBlock2D(starting_filters * 4, starting_filters * 4, 'duckv2', repeat=1)

        self.up2 = nn.Upsample(scale_factor=2, mode=self.interpolation)
        self.q6 = ConvBlock2D(starting_filters * 2, starting_filters * 2, 'duckv2', repeat=1)

        self.up1 = nn.Upsample(scale_factor=2, mode=self.interpolation)
        self.q1 = ConvBlock2D(starting_filters, starting_filters, 'duckv2', repeat=1)

        # Final upsampling and output
        self.up0 = nn.Upsample(scale_factor=2, mode=self.interpolation)
        self.z1 = ConvBlock2D(starting_filters, starting_filters, 'duckv2', repeat=1)
        
        self.output = nn.Conv2d(starting_filters, out_classes, kernel_size=1, stride=1)

    def forward(self, x):
        # Encoder pathway
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        
        # Initial conv block
        t0 = self.t0(x)
        
        # Intermediate blocks with additions
        s1 = self.l1i(t0) + p1
        t1 = self.t1(s1)
        
        s2 = self.l2i(t1) + p2
        t2 = self.t2(s2)
        
        s3 = self.l3i(t2) + p3
        t3 = self.t3(s3)
        
        s4 = self.l4i(t3) + p4
        t4 = self.t4(s4)
        
        s5 = self.l5i(t4) + p5
        t51 = self.t51(s5)
        t53 = self.t53(t51)
        
        # Decoder pathway
        l5o = self.up4(t53)
        c4 = l5o + t4
        q4 = self.q4(c4)
        
        l4o = self.up3(q4)
        c3 = l4o + t3
        q3 = self.q3(c3)
        
        l3o = self.up2(q3)
        c2 = l3o + t2
        q6 = self.q6(c2)
        
        l2o = self.up1(q6)
        c1 = l2o + t1
        q1 = self.q1(c1)
        
        l1o = self.up0(q1)
        c0 = l1o + t0
        z1 = self.z1(c0)
        
        output = torch.sigmoid(self.output(z1))
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock2D(nn.Module):
    def __init__(self, filters, block_type, repeat=1, dilation_rate=1, size=3, padding='same'):
        super(ConvBlock2D, self).__init__()
        self.filters = filters
        self.block_type = block_type
        self.repeat = repeat
        self.dilation_rate = dilation_rate
        self.size = size
        self.padding = padding

    def forward(self, x):
        result = x
        for _ in range(self.repeat):
            if self.block_type == 'separated':
                result = separated_conv2D_block(result, self.filters, self.size, self.padding)
            elif self.block_type == 'duckv2':
                result = duckv2_conv2D_block(result, self.filters, self.size)
            elif self.block_type == 'midscope':
                result = midscope_conv2D_block(result, self.filters)
            elif self.block_type == 'widescope':
                result = widescope_conv2D_block(result, self.filters)
            elif self.block_type == 'resnet':
                result = resnet_conv2D_block(result, self.filters, self.dilation_rate)
            elif self.block_type == 'conv':
                result = nn.Conv2d(x.size(1), self.filters, kernel_size=(self.size, self.size),
                                   padding=self.padding, dilation=self.dilation_rate)(result)
                result = F.relu(result)
            elif self.block_type == 'double_convolution':
                result = double_convolution_with_batch_normalization(result, self.filters, self.dilation_rate)
            else:
                return None
        return result

def separated_conv2D_block(x, filters, size=3, padding='same'):
    x = nn.Conv2d(x.size(1), filters, kernel_size=(1, size), padding=padding)(x)
    x = nn.BatchNorm2d(filters)(x)
    x = F.relu(x)
    x = nn.Conv2d(filters, filters, kernel_size=(size, 1), padding=padding)(x)
    x = nn.BatchNorm2d(filters)(x)
    return F.relu(x)

def duckv2_conv2D_block(x, filters, size=3):
    x = nn.BatchNorm2d(x.size(1))(x)
    x1 = widescope_conv2D_block(x, filters)
    x2 = midscope_conv2D_block(x, filters)
    x3 = resnet_conv2D_block(x, filters)
    x4 = resnet_conv2D_block(x, filters)
    x5 = resnet_conv2D_block(x, filters)
    x6 = separated_conv2D_block(x, filters, size=6)
    x = x1 + x2 + x3 + x4 + x5 + x6
    return nn.BatchNorm2d(x.size(1))(x)

def midscope_conv2D_block(x, filters):
    x = nn.Conv2d(x.size(1), filters, kernel_size=3, padding='same')(x)
    x = nn.BatchNorm2d(filters)(x)
    x = F.relu(x)
    x = nn.Conv2d(filters, filters, kernel_size=3, padding='same', dilation=2)(x)
    x = nn.BatchNorm2d(filters)(x)
    return F.relu(x)

def widescope_conv2D_block(x, filters):
    x = nn.Conv2d(x.size(1), filters, kernel_size=3, padding='same')(x)
    x = nn.BatchNorm2d(filters)(x)
    x = F.relu(x)
    x = nn.Conv2d(filters, filters, kernel_size=3, padding='same', dilation=2)(x)
    x = nn.BatchNorm2d(filters)(x)
    x = F.relu(x)
    x = nn.Conv2d(filters, filters, kernel_size=3, padding='same', dilation=3)(x)
    x = nn.BatchNorm2d(filters)(x)
    return F.relu(x)

def resnet_conv2D_block(x, filters, dilation_rate=1):
    x1 = nn.Conv2d(x.size(1), filters, kernel_size=1, padding='same', dilation=dilation_rate)(x)
    x = nn.Conv2d(x.size(1), filters, kernel_size=3, padding='same', dilation=dilation_rate)(x)
    x = nn.BatchNorm2d(filters)(x)
    x = F.relu(x)
    x = nn.Conv2d(filters, filters, kernel_size=3, padding='same', dilation=dilation_rate)(x)
    x = nn.BatchNorm2d(filters)(x)
    return F.relu(x + x1)

def double_convolution_with_batch_normalization(x, filters, dilation_rate=1):
    x = nn.Conv2d(x.size(1), filters, kernel_size=3, padding='same', dilation=dilation_rate)(x)
    x = nn.BatchNorm2d(filters)(x)
    x = F.relu(x)
    x = nn.Conv2d(filters, filters, kernel_size=3, padding='same', dilation=dilation_rate)(x)
    x = nn.BatchNorm2d(filters)(x)
    return F.relu(x)
