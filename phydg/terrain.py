"""Terrain derivatives from a DEM, computed on the fly with fixed convolutions."""
import torch
import torch.nn.functional as F

# Horn (1981) gradient kernels
_KX = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
_KY = _KX.t().contiguous()
_LAP = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])

TERRAIN_CHANNELS = ["rel_elev", "slope", "aspect_sin", "aspect_cos", "curvature", "roughness"]


def terrain_features(dem: torch.Tensor, pixel_size: float = 10.0) -> torch.Tensor:
    """dem [B, 1, H, W] in metres -> [B, 6, H, W] scale-free terrain features.

    Features are relative to the patch, so absolute elevation (which differs
    strongly between regions and is a shortcut for region identity) is removed.
    """
    x = F.pad(dem, (1, 1, 1, 1), mode="replicate")
    k = lambda w: w.to(dem).view(1, 1, 3, 3)
    dzdx = F.conv2d(x, k(_KX)) / pixel_size
    dzdy = F.conv2d(x, k(_KY)) / pixel_size
    slope = torch.atan(torch.sqrt(dzdx ** 2 + dzdy ** 2))  # radians
    aspect = torch.atan2(dzdy, -dzdx)
    curv = F.conv2d(x, k(_LAP)) / pixel_size ** 2
    rough = F.max_pool2d(x, 3, 1) + F.max_pool2d(-x, 3, 1)  # local relief (max - min)

    mu = dem.mean(dim=(2, 3), keepdim=True)
    sd = dem.std(dim=(2, 3), keepdim=True).clamp_min(1.0)
    return torch.cat([
        (dem - mu) / sd,
        slope / (torch.pi / 2),
        torch.sin(aspect) * torch.sin(slope),  # aspect is meaningless on flat ground
        torch.cos(aspect) * torch.sin(slope),
        torch.tanh(curv * 100.0),
        torch.tanh(rough / 50.0),
    ], dim=1)
