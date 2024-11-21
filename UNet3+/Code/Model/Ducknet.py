import torch
import torch.nn as nn
from config import WIDTH, HEIGHT

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

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
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

class DUCKNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, starting_filters=64, supervision=True):
        super(DUCKNet, self).__init__()
        
        self.supervision = supervision
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
        
        # Up-sampling layers
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=self.interpolation, align_corners=True),
            nn.Conv2d(starting_filters * 16, starting_filters * 8, kernel_size=1)
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

        # Decoder blocks
        self.q4 = ConvBlock2D(starting_filters * 8, starting_filters * 8, 'duckv2', repeat=1)
        self.q3 = ConvBlock2D(starting_filters * 4, starting_filters * 4, 'duckv2', repeat=1)
        self.q6 = ConvBlock2D(starting_filters * 2, starting_filters * 2, 'duckv2', repeat=1)
        self.q1 = ConvBlock2D(starting_filters, starting_filters, 'duckv2', repeat=1)

        # Final layers
        self.z1 = ConvBlock2D(starting_filters, starting_filters, 'duckv2', repeat=1)
        
        # Output layers for deep supervision
        self.output = nn.Conv2d(starting_filters, out_ch, kernel_size=1, stride=1)
        self.output2 = nn.Conv2d(starting_filters * 2, out_ch, kernel_size=1, stride=1)
        self.output3 = nn.Conv2d(starting_filters * 4, out_ch, kernel_size=1, stride=1)
        self.output4 = nn.Conv2d(starting_filters * 8, out_ch, kernel_size=1, stride=1)
        self.output5 = nn.Conv2d(starting_filters * 16, out_ch, kernel_size=1, stride=1)

        # Final upsampling to match target size
        self.final_upsample = nn.Upsample(size=(HEIGHT, WIDTH), mode='bilinear', align_corners=True)

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
        
        # Decoder pathway
        l5o = self.up4(t53)      # /16 -> /8
        c4 = l5o + t3
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
        
        # Deep supervision outputs
        d1 = self.final_upsample(self.output(z1))
        d2 = self.final_upsample(self.output2(q6))
        d3 = self.final_upsample(self.output3(q3))
        d4 = self.final_upsample(self.output4(q4))
        d5 = self.final_upsample(self.output5(t53))

        if self.supervision and self.training:
            return [d1, d2, d3, d4, d5]
        else:
            return d1