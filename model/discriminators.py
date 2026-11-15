import torch.nn as nn
import functools
import torch
import torch.nn.functional as F


def sample_valid_patches(real, fake, valid, size=128, count=2):
    """Identical uniform FOV sampling for real/fake; never conditioned on vessels."""
    if size > min(real.shape[-2:]) or size < 32 or count < 1:
        raise ValueError('local patches require 32 <= size <= image size and count >= 1')
    real_patches, fake_patches = [], []
    for i in range(real.shape[0]):
        # Integral images avoid an expensive dense size*size convolution.
        integral = F.pad(valid[i, 0].float(), (1, 0, 1, 0)).cumsum(0).cumsum(1)
        coverage = (integral[size:, size:] - integral[:-size, size:] -
                    integral[size:, :-size] + integral[:-size, :-size]) / (size * size)
        positions = (coverage >= 0.99).nonzero()
        if len(positions) == 0:
            raise ValueError('no local patch lies within the reference FOV; reduce patch_size')
        indices = torch.randint(len(positions), (count,), device=positions.device)
        for y, x in positions[indices].tolist():
            real_patches.append(real[i:i+1, :, y:y+size, x:x+size])
            fake_patches.append(fake[i:i+1, :, y:y+size, x:x+size])
    return torch.cat(real_patches), torch.cat(fake_patches)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
