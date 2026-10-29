"""Sen12Landslides S2 loader producing cloud-masked pre/post-event composites + DEM.

Reflectance is scaled physically (DN / 10000) rather than standardised with
dataset statistics, so no statistics from a held-out region leak into training.
"""
import ast
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .folds import INVENTORY_INDEX

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
# SCL classes treated as usable surface observations (excludes no-data, saturated,
# cloud shadow, medium/high cloud, cirrus).
SCL_VALID = (2, 4, 5, 6, 7, 11)


def parse_pre_post(attr, n_time: int) -> tuple[int, int, bool]:
    """Return (last_pre_idx, first_post_idx, date_known) from the pre_post_dates attribute.

    The preprocessing writes {"pre": i, "post": i+1}; pre == post marks an unknown
    or out-of-range event date. Multi-event patches store a list; the first valid
    event is used. Unknown dates fall back to splitting the series in half.
    """
    events = ast.literal_eval(attr) if isinstance(attr, str) else attr
    if isinstance(events, dict):
        events = [events]
    for ev in events or []:
        pre, post = int(ev["pre"]), int(ev["post"])
        if 0 <= pre < post < n_time:
            return pre, post, True
    half = n_time // 2
    return half - 1, half, False


def composite(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-pixel median over cloud-free frames. stack [T, C, H, W], valid [T, H, W]."""
    masked = np.where(valid[:, None], stack, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels -> fallback below
        out = np.nanmedian(masked, axis=0)
    fallback = np.median(stack, axis=0)
    return np.where(np.isnan(out), fallback, out).astype(np.float32)


def read_patch(path: str) -> dict:
    import xarray as xr

    with xr.open_dataset(path) as ds:
        ds = ds.sortby("time")
        yx = lambda v: v.transpose(..., "y", "x").values
        img = np.stack([yx(ds[b]) for b in S2_BANDS], axis=1).astype(np.float32)  # [T, C, H, W]
        scl = yx(ds["SCL"]) if "SCL" in ds else np.full(img[:, 0].shape, 4)
        dem = ds["DEM"]
        dem = yx(dem.isel(time=0) if "time" in dem.dims else dem).astype(np.float32)
        msk = ds["MASK"]
        msk = yx(msk.isel(time=0) if "time" in msk.dims else msk)
        pre, post, known = parse_pre_post(ds.attrs.get("pre_post_dates"), img.shape[0])

    img = np.clip(img, 0, 10000) / 10000.0
    valid = np.isin(scl, SCL_VALID)
    return {
        "pre": composite(img[: pre + 1], valid[: pre + 1]),
        "post": composite(img[post:], valid[post:]),
        "dem": dem[None],
        "msk": (msk > 0).astype(np.int64),
        "date_known": known,
    }


def augment(sample: dict, rng: np.random.Generator) -> dict:
    k = int(rng.integers(4))
    flip = bool(rng.integers(2))
    for key in ("pre", "post", "dem", "msk"):
        a = np.rot90(sample[key], k, axes=(-2, -1))
        sample[key] = np.ascontiguousarray(a[..., ::-1] if flip else a)
    return sample


class Sen12LS(Dataset):
    def __init__(self, entries: list[dict], root: str, train: bool = False, seed: int = 0):
        self.entries = entries
        self.root = Path(root)
        self.train = train
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_split(cls, split_json: str, part: str, root: str, **kw):
        with open(split_json) as f:
            return cls(json.load(f)[part], root, train=(part == "train"), **kw)

    @property
    def regions(self) -> list[int]:
        return [INVENTORY_INDEX[e["inventory"]] for e in self.entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        s = read_patch(str(self.root / e["s2"]))
        if self.train:
            s = augment(s, self.rng)
        out = {k: torch.from_numpy(s[k]) for k in ("pre", "post", "dem", "msk")}
        out["region"] = torch.tensor(INVENTORY_INDEX[e["inventory"]])
        out["date_known"] = torch.tensor(s["date_known"])
        return out


class RegionBalancedSampler(Sampler):
    """Draws `regions_per_batch` regions per batch and an equal share of patches from each.

    Balances the dominant Italian inventory (~36% of patches) and guarantees several
    regions per batch, which the cross-region invariance penalty needs.
    """

    def __init__(self, regions: list[int], batch_size: int, regions_per_batch: int = 4,
                 n_batches: int | None = None, seed: int = 0):
        self.by_region: dict[int, list[int]] = {}
        for i, r in enumerate(regions):
            self.by_region.setdefault(r, []).append(i)
        self.batch_size = batch_size
        self.rpb = min(regions_per_batch, len(self.by_region))
        self.n_batches = n_batches or len(regions) // batch_size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        keys = list(self.by_region)
        for _ in range(self.n_batches):
            chosen = self.rng.choice(keys, self.rpb, replace=False)
            per = np.full(self.rpb, self.batch_size // self.rpb)
            per[: self.batch_size % self.rpb] += 1
            yield [int(i) for r, n in zip(chosen, per)
                   for i in self.rng.choice(self.by_region[r], n, replace=len(self.by_region[r]) < n)]
