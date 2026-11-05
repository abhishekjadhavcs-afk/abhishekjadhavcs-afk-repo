from .phydg import PhyDGSlide
from .phydg2 import PhyDGUNet
from .unet import UNet

# capacity-matched control for PhyDGUNet (2.45M vs 2.46M parameters)
WIDE_WIDTHS = (36, 72, 144, 288)

V1_OPTS = ("style", "ibn", "film", "poe")  # v1: all on by default, --no-<opt> ablates
V2_OFF = ("change", "film")                 # v2: on by default, --no-<opt> ablates
V2_ON = ("poe", "style", "ibn")             # v2: off by default, --<opt> adds


def build_model(name: str, **kw):
    if name == "unet":
        return UNet()
    if name == "unet-wide":
        return UNet(WIDE_WIDTHS)
    if name == "phydg":
        return PhyDGSlide(**kw)
    if name == "phydg2":
        return PhyDGUNet(**kw)
    raise KeyError(name)


def model_spec(args) -> tuple[dict, float, str]:
    """(model kwargs, V-REx weight, run tag) from parsed train.py arguments (a Namespace or dict).

    Shared by train.py, run_grid.py and eval_thresholds.py so a run's folder name and its
    model are always derived the same way. Tags of earlier runs are unchanged.
    """
    a = args if isinstance(args, dict) else vars(args)
    get = lambda k, d=False: a.get(k, d)  # noqa: E731  results.json from older runs lacks v2 keys
    name = a["model"]
    vrex = a.get("vrex")
    if name in ("unet", "unet-wide"):
        kw, tag = {}, name
        vrex = 0.0 if vrex is None else vrex
    elif name == "phydg":
        kw = {c: not get(f"no_{c}") for c in V1_OPTS}
        tag = "phydg" + "".join(f"-no{c}" for c in V1_OPTS if get(f"no_{c}"))
        vrex = 1.0 if vrex is None else vrex
    elif name == "phydg2":
        kw = {c: not get(f"no_{c}") for c in V2_OFF} | {c: bool(get(c)) for c in V2_ON}
        tag = "phydg2" + "".join(f"-no{c}" for c in V2_OFF if get(f"no_{c}")) \
            + "".join(f"+{c}" for c in V2_ON if get(c))
        vrex = 0.0 if vrex is None else vrex
    else:
        raise KeyError(name)
    return kw, float(vrex), f"{tag}_vrex{float(vrex):g}"
