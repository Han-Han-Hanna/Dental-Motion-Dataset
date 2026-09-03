import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class U_Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=9, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.center = ConvBlock(base * 8, base * 16)
        self.d4 = UpBlock(base * 16, base * 8, base * 8)
        self.d3 = UpBlock(base * 8, base * 4, base * 4)
        self.d2 = UpBlock(base * 4, base * 2, base * 2)
        self.d1 = UpBlock(base * 2, base, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        center = self.center(self.pool(e4))
        d4 = self.d4(center, e4)
        d3 = self.d3(d4, e3)
        d2 = self.d2(d3, e2)
        d1 = self.d1(d2, e1)
        return self.out(d1)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, hidden_channels):
        super().__init__()
        self.gate = nn.Conv2d(gate_channels, hidden_channels, 1, bias=False)
        self.skip = nn.Conv2d(skip_channels, hidden_channels, 1, bias=False)
        self.score = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return skip * self.score(self.gate(gate) + self.skip(skip))


class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.attention = AttentionGate(out_channels, skip_channels, max(out_channels // 2, 16))
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        skip = self.attention(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class AttU_Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=9, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.center = ConvBlock(base * 8, base * 16)
        self.d4 = AttentionUpBlock(base * 16, base * 8, base * 8)
        self.d3 = AttentionUpBlock(base * 8, base * 4, base * 4)
        self.d2 = AttentionUpBlock(base * 4, base * 2, base * 2)
        self.d1 = AttentionUpBlock(base * 2, base, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        center = self.center(self.pool(e4))
        d4 = self.d4(center, e4)
        d3 = self.d3(d4, e3)
        d2 = self.d2(d3, e2)
        d1 = self.d1(d2, e1)
        return self.out(d1)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=9, base=32):
        super().__init__()
        channels = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2)
        self.x00 = ConvBlock(in_channels, channels[0])
        self.x10 = ConvBlock(channels[0], channels[1])
        self.x20 = ConvBlock(channels[1], channels[2])
        self.x30 = ConvBlock(channels[2], channels[3])
        self.x40 = ConvBlock(channels[3], channels[4])
        self.x01 = ConvBlock(channels[0] + channels[1], channels[0])
        self.x11 = ConvBlock(channels[1] + channels[2], channels[1])
        self.x21 = ConvBlock(channels[2] + channels[3], channels[2])
        self.x31 = ConvBlock(channels[3] + channels[4], channels[3])
        self.x02 = ConvBlock(channels[0] * 2 + channels[1], channels[0])
        self.x12 = ConvBlock(channels[1] * 2 + channels[2], channels[1])
        self.x22 = ConvBlock(channels[2] * 2 + channels[3], channels[2])
        self.x03 = ConvBlock(channels[0] * 3 + channels[1], channels[0])
        self.x13 = ConvBlock(channels[1] * 3 + channels[2], channels[1])
        self.x04 = ConvBlock(channels[0] * 4 + channels[1], channels[0])
        self.out = nn.Conv2d(channels[0], num_classes, 1)

    def forward(self, x):
        up = lambda value, ref: F.interpolate(value, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))
        x01 = self.x01(torch.cat([x00, up(x10, x00)], 1))
        x11 = self.x11(torch.cat([x10, up(x20, x10)], 1))
        x21 = self.x21(torch.cat([x20, up(x30, x20)], 1))
        x31 = self.x31(torch.cat([x30, up(x40, x30)], 1))
        x02 = self.x02(torch.cat([x00, x01, up(x11, x00)], 1))
        x12 = self.x12(torch.cat([x10, x11, up(x21, x10)], 1))
        x22 = self.x22(torch.cat([x20, x21, up(x31, x20)], 1))
        x03 = self.x03(torch.cat([x00, x01, x02, up(x12, x00)], 1))
        x13 = self.x13(torch.cat([x10, x11, x12, up(x22, x10)], 1))
        x04 = self.x04(torch.cat([x00, x01, x02, x03, up(x13, x00)], 1))
        return self.out(x04)


class FullScaleProject(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, size):
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return self.block(x)


class UNet3Plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=9, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.e5 = ConvBlock(base * 8, base * 16)
        self.p4 = nn.ModuleList([FullScaleProject(c, base) for c in [base, base * 2, base * 4, base * 8, base * 16]])
        self.d4 = ConvBlock(base * 5, base * 5)
        self.p3 = nn.ModuleList([FullScaleProject(c, base) for c in [base, base * 2, base * 4, base * 5, base * 16]])
        self.d3 = ConvBlock(base * 5, base * 5)
        self.p2 = nn.ModuleList([FullScaleProject(c, base) for c in [base, base * 2, base * 5, base * 5, base * 16]])
        self.d2 = ConvBlock(base * 5, base * 5)
        self.p1 = nn.ModuleList([FullScaleProject(c, base) for c in [base, base * 5, base * 5, base * 5, base * 16]])
        self.d1 = ConvBlock(base * 5, base * 5)
        self.out = nn.Conv2d(base * 5, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        e5 = self.e5(self.pool(e4))
        s4 = e4.shape[-2:]
        d4 = self.d4(torch.cat([layer(value, s4) for layer, value in zip(self.p4, [e1, e2, e3, e4, e5])], 1))
        s3 = e3.shape[-2:]
        d3 = self.d3(torch.cat([layer(value, s3) for layer, value in zip(self.p3, [e1, e2, e3, d4, e5])], 1))
        s2 = e2.shape[-2:]
        d2 = self.d2(torch.cat([layer(value, s2) for layer, value in zip(self.p2, [e1, e2, d3, d4, e5])], 1))
        s1 = e1.shape[-2:]
        d1 = self.d1(torch.cat([layer(value, s1) for layer, value in zip(self.p1, [e1, d2, d3, d4, e5])], 1))
        return self.out(d1)


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 8, 8)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.block(x)


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch = out_channels // 4
        self.b1 = nn.Conv2d(in_channels, branch, 1)
        self.b2 = nn.Conv2d(in_channels, branch, 3, padding=1)
        self.b3 = nn.Conv2d(in_channels, branch, 3, padding=2, dilation=2)
        self.b4 = nn.Conv2d(in_channels, out_channels - branch * 3, 5, padding=2)
        self.mix = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), ChannelAttention(out_channels))

    def forward(self, x):
        return self.mix(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1))


class CMUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=9, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = MultiScaleBlock(in_channels, base)
        self.e2 = MultiScaleBlock(base, base * 2)
        self.e3 = MultiScaleBlock(base * 2, base * 4)
        self.e4 = MultiScaleBlock(base * 4, base * 8)
        self.center = MultiScaleBlock(base * 8, base * 16)
        self.d4 = UpBlock(base * 16, base * 8, base * 8)
        self.d3 = UpBlock(base * 8, base * 4, base * 4)
        self.d2 = UpBlock(base * 4, base * 2, base * 2)
        self.d1 = UpBlock(base * 2, base, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        center = self.center(self.pool(e4))
        return self.out(self.d1(self.d2(self.d3(self.d4(center, e4), e3), e2), e1))


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.expand = nn.Conv2d(channels, channels * 4, 1)
        self.reduce = nn.Conv2d(channels * 4, channels, 1)
        self.scale = nn.Parameter(torch.ones(channels) * 1e-6)

    def forward(self, x):
        value = self.depthwise(x)
        value = self.norm(value)
        value = self.reduce(F.gelu(self.expand(value)))
        return x + value * self.scale.view(1, -1, 1, 1)


class CMUNeXt(nn.Module):
    def __init__(self, in_channels=3, num_classes=9, base=32):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, base, 3, padding=1)
        self.e1 = nn.Sequential(ConvNeXtBlock(base), ConvNeXtBlock(base))
        self.down1 = nn.Conv2d(base, base * 2, 2, stride=2)
        self.e2 = nn.Sequential(ConvNeXtBlock(base * 2), ConvNeXtBlock(base * 2))
        self.down2 = nn.Conv2d(base * 2, base * 4, 2, stride=2)
        self.e3 = nn.Sequential(ConvNeXtBlock(base * 4), ConvNeXtBlock(base * 4))
        self.down3 = nn.Conv2d(base * 4, base * 8, 2, stride=2)
        self.e4 = nn.Sequential(ConvNeXtBlock(base * 8), ConvNeXtBlock(base * 8))
        self.down4 = nn.Conv2d(base * 8, base * 16, 2, stride=2)
        self.center = nn.Sequential(ConvNeXtBlock(base * 16), ConvNeXtBlock(base * 16))
        self.d4 = UpBlock(base * 16, base * 8, base * 8)
        self.d3 = UpBlock(base * 8, base * 4, base * 4)
        self.d2 = UpBlock(base * 4, base * 2, base * 2)
        self.d1 = UpBlock(base * 2, base, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(self.stem(x))
        e2 = self.e2(self.down1(e1))
        e3 = self.e3(self.down2(e2))
        e4 = self.e4(self.down3(e3))
        center = self.center(self.down4(e4))
        return self.out(self.d1(self.d2(self.d3(self.d4(center, e4), e3), e2), e1))


def build_segmentation_model(name, num_classes=9):
    models = {
        'AttU-Net': AttU_Net,
        'CMU-Net': CMUNet,
        'CMUNeXt': CMUNeXt,
        'U-Net': U_Net,
        'U-Net++': UNetPlusPlus,
        'UNet 3+': UNet3Plus,
    }
    return models[name](num_classes=num_classes)
