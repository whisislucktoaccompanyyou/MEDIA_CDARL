"""Predict Lab a/b from a frozen SPADE feature. Does not change luminance."""

import torch
import torch.nn as nn


class ColorHead(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 2, kernel_size=3),
            nn.Tanh(),
        )
        # Start at gray. A random head would paint the first previews.
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, feature):
        # OpenCV Lab a/b are stored as 0-255 with 128 at neutral.
        return (self.net(feature) + 1.0) * 0.5 * 255.0
