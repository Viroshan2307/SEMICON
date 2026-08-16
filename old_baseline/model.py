"""
model.py

Model architecture for joint image denoising + 2x super-resolution.
Input:  (B, 1, 128, 128) noisy, low-resolution grayscale image
Output: (B, 1, 256, 256) clean, full-resolution grayscale image

This file is imported by both train.py and evaluate.py, so the
architecture only needs to be defined once and stays consistent
between training and evaluation.
"""

import torch.nn as nn


class ResidualBlock(nn.Module):
    """A small residual block: helps the network learn to remove noise
    without losing the underlying signal (skip connection preserves detail)."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual  # skip connection


class RestorationNet(nn.Module):
    """
    Joint denoising + 2x super-resolution network.

    Design:
    1. Feature extraction at input resolution
    2. Residual blocks refine features (this is where denoising happens)
    3. PixelShuffle upsamples 128->256 while reconstructing detail
       (PixelShuffle is preferred over plain upsampling/deconv because
       it avoids the checkerboard artifacts those methods can introduce)
    4. Final conv maps back to a single grayscale channel
    """

    def __init__(self, base_channels=64, num_res_blocks=6):
        super().__init__()

        self.entry = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_res_blocks)]
        )

        self.bridge = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.exit = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        feat = self.entry(x)
        res = self.res_blocks(feat)
        res = self.bridge(res)
        feat = feat + res  # global skip connection: preserves low-level detail
        up = self.upsample(feat)
        out = self.exit(up)
        return out
