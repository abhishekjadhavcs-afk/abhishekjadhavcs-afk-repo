"""PhyDG-Slide: physics-anchored spectro-temporal network for unseen-region landslide mapping.

Components (each can be switched off for ablations):
  style    - SpectralStyle: train-time cross-region style randomisation in reflectance space
  ibn      - instance/batch norm in the shallow change-encoder stages (style removal)
  film     - terrain encoder modulates change features at every scale (FiLM)
  poe      - product of experts: final logit = change logit + terrain-plausibility logit
The cross-region invariance penalty (V-REx) lives in losses.py.
"""
import torch
import torch.nn as nn

from ..spectral import N_CHANGE_CHANNELS, SpectralStyle, change_features
from ..terrain import TERRAIN_CHANNELS, terrain_features
from .unet import Decoder, conv_block


class TerrainEncoder(nn.Module):
    def __init__(self, widths):
        super().__init__()
        cin = len(TERRAIN_CHANNELS)
        self.blocks = nn.ModuleList()
        for w in widths:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(cin, w // 2, 3, padding=1), nn.GroupNorm(4, w // 2), nn.ReLU(inplace=True)))
            cin = w // 2
        self.pool = nn.MaxPool2d(2)
        # zero-initialised FiLM heads: the network starts as an unmodulated change encoder
        self.film = nn.ModuleList(nn.Conv2d(w // 2, 2 * w, 1) for w in widths)
        for f in self.film:
            nn.init.zeros_(f.weight)
            nn.init.zeros_(f.bias)
        self.prior = nn.Conv2d(widths[0] // 2, 1, 1)

    def forward(self, t):
        feats, films = [], []
        for i, (block, film) in enumerate(zip(self.blocks, self.film)):
            t = block(t if i == 0 else self.pool(t))
            feats.append(t)
            films.append(film(t).chunk(2, dim=1))
        return films, self.prior(feats[0])


class PhyDGSlide(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), style=True, ibn=True, film=True, poe=True):
        super().__init__()
        self.use_film, self.use_poe = film, poe
        self.style = SpectralStyle() if style else None
        cin = N_CHANGE_CHANNELS
        self.enc = nn.ModuleList()
        for i, w in enumerate(widths):
            self.enc.append(conv_block(cin, w, norm="ibn" if ibn and i < 2 else "bn"))
            cin = w
        self.pool = nn.MaxPool2d(2)
        self.terrain = TerrainEncoder(widths)
        self.dec = Decoder(widths)

    def forward(self, pre, post, dem, region=None):
        if self.style is not None and region is not None:
            pre, post = self.style(pre, post, region)
        x = change_features(pre, post)
        films, prior = self.terrain(terrain_features(dem))
        feats = []
        for i, (block, (gamma, beta)) in enumerate(zip(self.enc, films)):
            x = block(x if i == 0 else self.pool(x))
            if self.use_film:
                x = x * (1 + gamma) + beta
            feats.append(x)
        change_logits = self.dec(feats)
        out = {"change_logits": change_logits, "prior_logits": prior}
        out["logits"] = change_logits + prior if self.use_poe else change_logits
        return out
