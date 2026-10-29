"""Synthetic NetCDF patches in the Sen12Landslides layout, for pipeline tests only.

These contain random data and a fake "landslide" (vegetation loss on a slope); they
exist to exercise the code path, not to produce any reportable result.
"""
import numpy as np
import pandas as pd
import xarray as xr

from phydg.data import S2_BANDS


def write_fake_patch(path, rng, size=64, n_time=15, event_idx=7):
    yy, xx = np.mgrid[:size, :size]
    dem = (500 + 8 * yy + 20 * np.sin(xx / 7)).astype(np.int16)
    mask = ((yy - size / 2) ** 2 / 60 + (xx - size / 2) ** 2 / 20 < 4).astype(np.uint8)
    base = rng.uniform(300, 4000, size=(len(S2_BANDS), 1, 1))
    data = {}
    for bi, b in enumerate(S2_BANDS):
        arr = base[bi] + rng.normal(0, 100, size=(n_time, size, size))
        if b in ("B08", "B8A"):  # NIR drops after the event inside the scar
            arr[event_idx + 1:] -= 1500.0 * mask
        data[b] = (("time", "y", "x"), np.clip(arr, 0, 10000).astype(np.int16))
    scl = np.full((n_time, size, size), 4, np.int16)
    scl[2, :10] = 9  # a cloud
    data["SCL"] = (("time", "y", "x"), scl)
    data["DEM"] = (("time", "y", "x"), np.broadcast_to(dem, (n_time, size, size)).copy())
    data["MASK"] = (("time", "y", "x"), np.broadcast_to(mask, (n_time, size, size)).copy())
    times = pd.date_range("2022-01-01", periods=n_time, freq="24D")
    ds = xr.Dataset(data, coords={"time": times, "y": np.arange(size), "x": np.arange(size)})
    ds.attrs["pre_post_dates"] = str({"pre": event_idx, "post": event_idx + 1})
    ds.to_netcdf(path)


def make_fake_dataset(root, inventories=("italy", "hiroshima", "chimanimani", "newzealand"), per_inv=6, seed=0):
    rng = np.random.default_rng(seed)
    (root / "s2").mkdir(parents=True, exist_ok=True)
    entries = []
    for inv in inventories:
        for i in range(per_inv):
            rel = f"s2/{inv}_s2_{i}.nc"
            write_fake_patch(root / rel, rng)
            entries.append({"id": f"{inv}_{i}", "s2": rel, "inventory": inv, "pixel_annotated": 0})
    return entries
