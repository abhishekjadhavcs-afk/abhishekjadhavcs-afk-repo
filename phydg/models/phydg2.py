"""PhyDG-Slide v2: physics branches added on top of an unchanged U-Net.

The base is exactly models.unet.UNet (same 26-channel input, encoder and decoder).
Two branches plug in after each of the four encoder blocks, before the skip
connection and pooling, so both the decoder and the deeper stages see them:

  (A) change branch   x <- x + Z_s(c_s)            Z_s: 1x1 conv, zero-initialised
  (B) terrain FiLM    x <- x * (1 + gamma_s) + beta_s  gamma, beta heads zero-initialised

Because every connection into the base starts at zero, the untrained model
computes exactly the baseline U-Net; whatever differs was learned.
The terrain branch also has an auxiliary plausibility head (B'), trained with a
small loss and used for interpretation; it enters the prediction only with poe=True.

Options (defaults are the configuration supported by the ablations so far):
  change=True, film=True, poe=False, style=False (weakened if on), ibn=False.
"""
import torch
import torch.nn as nn

from ..spectral import N_CHANGE_CHANNELS, SpectralStyle, change_features
from ..terrain import terrain_features
from .phydg import TerrainEncoder
from .unet import UNet, conv_block


class ChangeEncoder(nn.Module):
    """Light encoder for explicit change evidence; one zero-initialised 1x1 projection per scale."""

    def __init__(self, base_widths, ratio=0.5):
        super().__init__()
        widths = [max(8, int(w * ratio)) for w in base_widths]
        self.blocks = nn.ModuleList()
        cin = N_CHANGE_CHANNELS
        for w in widths:
            self.blocks.append(conv_block(cin, w))
            cin = w
        self.pool = nn.MaxPool2d(2)
        self.proj = nn.ModuleList(nn.Conv2d(w, bw, 1) for w, bw in zip(widths, base_widths))
        for p in self.proj:
            nn.init.zeros_(p.weight)
            nn.init.zeros_(p.bias)

    def forward(self, c):
        out = []
        for i, (block, proj) in enumerate(zip(self.blocks, self.proj)):
            c = block(c if i == 0 else self.pool(c))
            out.append(proj(c))
        return out


class PhyDGUNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), change=True, film=True, poe=False,
                 style=False, ibn=False):
        super().__init__()
        self.use_change, self.use_film, self.use_poe = change, film, poe
        self.base = UNet(widths, ibn_stages=2 if ibn else 0)
        self.change = ChangeEncoder(widths) if change else None
        self.terrain = TerrainEncoder(widths)  # always built: provides the auxiliary prior
        # weakened relative to v1 (p 0.8 -> 0.5, gain 0.15 -> 0.05, keeps >= 80% of own style)
        self.style = SpectralStyle(p=0.5, gain_sigma=0.05, offset_sigma=0.005, mix_min=0.8) if style else None

    def forward(self, pre, post, dem, region=None):
        if self.style is not None and region is not None:
            pre, post = self.style(pre, post, region)
        t = terrain_features(dem)
        x = torch.cat([pre, post, t], dim=1)  # the baseline U-Net's input, unchanged
        films, prior = self.terrain(t)
        injections = self.change(change_features(pre, post)) if self.use_change else [None] * len(films)

        feats = []
        for i, (block, inj, (gamma, beta)) in enumerate(zip(self.base.enc, injections, films)):
            x = block(x if i == 0 else self.base.pool(x))
            if inj is not None:
                x = x + inj
            if self.use_film:
                x = x * (1 + gamma) + beta
            feats.append(x)
        logits = self.base.dec(feats)
        out = {"base_logits": logits, "prior_logits": prior}
        out["logits"] = logits + prior if self.use_poe else logits
        return out
