"""
teammate_model.py

Architecture from teammate's script: a NAFNet-style U-Net with channel
attention, encoder-decoder structure, and PixelShuffle upsampling.
Extracted into its own file so it can be loaded independently for
verification against our own evaluation pipeline.
"""

import torch.nn as nn


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(c * 2, c * 2, kernel_size=3, padding=1, groups=c * 2)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(c, c, kernel_size=1)
        self.norm = nn.GroupNorm(1, c)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1)
        )

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv2(x)
        return res + x


class HighResSemiconductorNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_dim=64, scale_factor=2):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        self.enc1 = nn.Sequential(NAFBlock(base_dim), NAFBlock(base_dim))
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(NAFBlock(base_dim * 2), NAFBlock(base_dim * 2))
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(NAFBlock(base_dim * 4), NAFBlock(base_dim * 4))

        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(NAFBlock(base_dim * 2), NAFBlock(base_dim * 2))

        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(NAFBlock(base_dim), NAFBlock(base_dim))

        self.upsample_head = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x_in = self.in_conv(x)
        e1 = self.enc1(x_in)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        d2 = self.dec2(self.up2(b) + e2)
        d1 = self.dec1(self.up1(d2) + e1)
        return self.upsample_head(d1)
