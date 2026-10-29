"""Spectral indices, change features and region-style randomisation for S2 composites.

Band order follows data.S2_BANDS: B02 B03 B04 B05 B06 B07 B08 B8A B11 B12.
"""
import torch
import torch.nn as nn

B02, B03, B04, B08, B11, B12 = 0, 1, 2, 6, 8, 9


def _nd(a, b):
    return (a - b) / (a + b).clamp_min(1e-4)


def indices(x: torch.Tensor) -> torch.Tensor:
    """x [B, 10, H, W] reflectance -> [B, 4, H, W]: NDVI, NBR, BSI, NDWI."""
    b = lambda i: x[:, i:i + 1]
    return torch.cat([
        _nd(b(B08), b(B04)),
        _nd(b(B08), b(B12)),
        _nd(b(B11) + b(B04), b(B08) + b(B02)),
        _nd(b(B03), b(B08)),
    ], dim=1)


N_CHANGE_CHANNELS = 10 + 10 + 12


def change_features(pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
    """[post - pre, post, idx(pre), idx(post), idx(post) - idx(pre)] -> [B, 32, H, W]."""
    ip, iq = indices(pre), indices(post)
    return torch.cat([post - pre, post, ip, iq, iq - ip], dim=1)


class SpectralStyle(nn.Module):
    """Train-time randomisation of region "style" in reflectance space.

    Two perturbations, both applied identically to the pre and post composite so
    the event-driven change survives while absolute appearance (vegetation type,
    soil colour, haze, phenology) is randomised:
      1. random per-band gain/offset (smooth across neighbouring bands), and
      2. cross-region style mixing: per-band mean/std of a patch are mixed with
         those of a patch from a *different* region in the batch (MixStyle-like,
         but at the input and domain-aware).
    """

    def __init__(self, p: float = 0.8, gain_sigma: float = 0.15, offset_sigma: float = 0.02,
                 mix_alpha: float = 0.3):
        super().__init__()
        self.p, self.gs, self.os = p, gain_sigma, offset_sigma
        self.beta = torch.distributions.Beta(mix_alpha, mix_alpha)

    def forward(self, pre, post, region):
        if not self.training:
            return pre, post
        B, C = pre.shape[:2]
        dev = pre.device
        apply = (torch.rand(B, 1, 1, 1, device=dev) < self.p).float()

        # 1. smooth random spectral gain/offset
        knots = torch.randn(B, 1, 4, device=dev)
        smooth = nn.functional.interpolate(knots, size=C, mode="linear", align_corners=True)
        gain = torch.exp(self.gs * smooth).view(B, C, 1, 1)
        offset = (self.os * torch.randn(B, 1, 1, 1, device=dev)).expand(B, C, 1, 1)

        # 2. cross-region statistics mixing
        both = torch.cat([pre, post], dim=-1)
        mu = both.mean(dim=(2, 3), keepdim=True)
        sd = both.std(dim=(2, 3), keepdim=True) + 1e-5
        perm = self._other_region_perm(region)
        lam = self.beta.sample((B, 1, 1, 1)).to(dev)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sd_mix = lam * sd + (1 - lam) * sd[perm]

        def restyle(x):
            x = (x - mu) / sd * sd_mix + mu_mix
            x = x * gain + offset
            return x.clamp_min(0.0)

        pre = apply * restyle(pre) + (1 - apply) * pre
        post = apply * restyle(post) + (1 - apply) * post
        return pre, post

    @staticmethod
    def _other_region_perm(region):
        """For each sample, index of a random sample from another region (self if none)."""
        B = region.shape[0]
        diff = region[None, :] != region[:, None]
        scores = torch.rand(B, B, device=region.device) * diff
        perm = scores.argmax(dim=1)
        return torch.where(diff.any(dim=1), perm, torch.arange(B, device=region.device))
