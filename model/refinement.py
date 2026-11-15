"""Shared, geometry-conditioned residual refinement (not a diffusion sampler)."""
import torch
from torch import nn
from torch.nn import functional as F


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1),
                         nn.GroupNorm(8, outputs), nn.SiLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1),
                         nn.GroupNorm(8, outputs), nn.SiLU())


class StructureRefiner(nn.Module):
    def __init__(self, channels=32, steps=3, alpha=0.1):
        super().__init__()
        if steps < 1 or channels % 8 or alpha <= 0:
            raise ValueError('steps >= 1, channels divisible by 8, alpha > 0 required')
        self.steps, self.alpha = int(steps), float(alpha)
        self.enc0 = block(8, channels)
        self.enc1 = block(channels + 4, channels * 2)
        self.bottom = block(channels * 2 + 4, channels * 4)
        self.dec1 = block(channels * 6 + 4, channels * 2)
        self.dec0 = block(channels * 3 + 4, channels)
        self.head = nn.Conv2d(channels, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def residual(self, image, initial, low, geometry, step):
        resize = lambda x, shape: F.interpolate(x, size=shape, mode='bilinear', align_corners=False)
        e0 = self.enc0(torch.cat([image, initial, low, geometry, torch.full_like(image, step)], 1))
        p0 = F.avg_pool2d(e0, 2)
        e1 = self.enc1(torch.cat([p0, resize(geometry, p0.shape[-2:])], 1))
        p1 = F.avg_pool2d(e1, 2)
        bottom = self.bottom(torch.cat([p1, resize(geometry, p1.shape[-2:])], 1))
        d1 = self.dec1(torch.cat([resize(bottom, e1.shape[-2:]), e1, resize(geometry, e1.shape[-2:])], 1))
        d0 = self.dec0(torch.cat([resize(d1, e0.shape[-2:]), e0, geometry], 1))
        return self.head(d0)

    def forward(self, initial, reference, geometry, steps=None):
        steps = self.steps if steps is None else int(steps)
        if steps < 0:
            raise ValueError('refinement steps cannot be negative')
        # A fixed coarse grid discards reference vessel edges and preserves illumination.
        coarse = F.adaptive_avg_pool2d(reference, (8, 8))
        low = F.interpolate(coarse, size=reference.shape[-2:], mode='bilinear', align_corners=False)
        images = [initial]
        logits = torch.atanh(initial.clamp(-1 + 1e-6, 1 - 1e-6))
        for k in range(steps):
            logits = logits + self.alpha * torch.tanh(self.residual(images[-1], initial, low, geometry, (k + 1) / steps))
            images.append(torch.tanh(logits))
        return images
