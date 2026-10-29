"""Baseline: 2D U-Net on early-fused [pre, post, terrain] composites."""
import torch
import torch.nn as nn

from ..terrain import TERRAIN_CHANNELS, terrain_features


def conv_block(cin, cout, norm="bn"):
    def n(c):
        if norm == "ibn":  # half instance norm, half batch norm (IBN-a)
            return IBN(c)
        return nn.BatchNorm2d(c)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), n(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class IBN(nn.Module):
    """Instance norm on half the channels removes per-patch style; batch norm keeps content."""

    def __init__(self, c):
        super().__init__()
        self.half = c // 2
        self.inorm = nn.InstanceNorm2d(self.half, affine=True)
        self.bnorm = nn.BatchNorm2d(c - self.half)

    def forward(self, x):
        a, b = x[:, :self.half], x[:, self.half:]
        return torch.cat([self.inorm(a), self.bnorm(b)], dim=1)


class Decoder(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for hi, lo in zip(widths[:0:-1], widths[-2::-1]):
            self.ups.append(nn.ConvTranspose2d(hi, lo, 2, stride=2))
            self.blocks.append(conv_block(2 * lo, lo))
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, feats):
        x = feats[-1]
        for up, block, skip in zip(self.ups, self.blocks, feats[-2::-1]):
            x = block(torch.cat([up(x), skip], dim=1))
        return self.head(x)


class UNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256)):
        super().__init__()
        cin = 10 + 10 + len(TERRAIN_CHANNELS)
        self.enc = nn.ModuleList()
        for w in widths:
            self.enc.append(conv_block(cin, w))
            cin = w
        self.pool = nn.MaxPool2d(2)
        self.dec = Decoder(widths)

    def forward(self, pre, post, dem, region=None):
        x = torch.cat([pre, post, terrain_features(dem)], dim=1)
        feats = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else self.pool(x))
            feats.append(x)
        return {"logits": self.dec(feats)}
