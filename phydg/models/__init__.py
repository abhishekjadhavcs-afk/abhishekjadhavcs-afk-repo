from .phydg import PhyDGSlide
from .unet import UNet


def build_model(name: str, **kw):
    if name == "unet":
        return UNet()
    if name == "phydg":
        return PhyDGSlide(**kw)
    raise KeyError(name)
