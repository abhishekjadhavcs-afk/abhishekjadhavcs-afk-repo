import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures import make_fake_dataset  # noqa: E402
from phydg.data import RegionBalancedSampler, Sen12LS, parse_pre_post  # noqa: E402
from phydg.folds import CLUSTERS, INVENTORY_TO_CLUSTER  # noqa: E402
from phydg.losses import total_loss, vrex_penalty  # noqa: E402
from phydg.models import build_model  # noqa: E402
from phydg.spectral import SpectralStyle  # noqa: E402
from phydg.splits import load_entries, loco_split  # noqa: E402
from phydg.terrain import terrain_features  # noqa: E402

OFFICIAL = ROOT / "splits/official_s2/splits.json"


@pytest.mark.parametrize("cluster", sorted(CLUSTERS))
def test_loco_split_has_no_region_leakage(cluster):
    entries = load_entries(OFFICIAL)
    s = loco_split(entries, cluster)
    test_inv = {e["inventory"] for e in s["test"]}
    train_inv = {e["inventory"] for e in s["train"] + s["val"]}
    assert test_inv == set(CLUSTERS[cluster]) & {e["inventory"] for e in entries}
    assert not test_inv & train_inv
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == len(entries)
    assert len({e["id"] for part in s.values() for e in part}) == len(entries)


def test_every_official_inventory_has_a_cluster():
    assert {e["inventory"] for e in load_entries(OFFICIAL)} <= set(INVENTORY_TO_CLUSTER)


def test_parse_pre_post():
    assert parse_pre_post("{'pre': 7, 'post': 8}", 15) == (7, 8, True)
    assert parse_pre_post("[{'pre': 0, 'post': 0}, {'pre': 3, 'post': 4}]", 15) == (3, 4, True)
    assert parse_pre_post("{'pre': 0, 'post': 0}", 15) == (6, 7, False)
    assert parse_pre_post(None, 15) == (6, 7, False)


def test_terrain_features_flat_and_slope():
    flat = torch.full((1, 1, 16, 16), 100.0)
    f = terrain_features(flat)
    assert f.shape == (1, 6, 16, 16) and torch.allclose(f[:, 1], torch.zeros(1, 16, 16))
    ramp = torch.arange(16.0).view(1, 1, 16, 1).expand(1, 1, 16, 16) * 10.0  # 45 degrees at 10 m
    s = terrain_features(ramp)[0, 1, 4:12, 4:12]
    assert torch.allclose(s, torch.full_like(s, 0.5), atol=1e-4)


def test_spectral_style_preserves_change_sign_and_is_identity_in_eval():
    torch.manual_seed(0)
    style = SpectralStyle(p=1.0)
    pre = torch.rand(4, 10, 8, 8) * 0.3 + 0.2
    post = pre.clone()
    post[:, 6] -= 0.15  # NIR loss
    region = torch.tensor([0, 0, 1, 2])
    a, b = style(pre, post, region)
    assert ((b - a)[:, 6] < 0).float().mean() > 0.95
    style.eval()
    a, b = style(pre, post, region)
    assert torch.equal(a, pre) and torch.equal(b, post)


def test_vrex_zero_when_risks_equal():
    losses = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert vrex_penalty(losses, torch.tensor([0, 0, 1, 1])) == 0
    assert vrex_penalty(torch.tensor([1.0, 1.0, 3.0, 3.0]), torch.tensor([0, 0, 1, 1])) == 1.0


def test_sampler_mixes_regions():
    regions = [0] * 100 + [1] * 5 + [2] * 5 + [3] * 5
    batches = list(RegionBalancedSampler(regions, 8, 4, n_batches=20))
    for b in batches:
        assert len(b) == 8 and len({regions[i] for i in b}) == 4


@pytest.mark.parametrize("name,kw", [("unet", {}), ("phydg", {}),
                                     ("phydg", {"style": False, "ibn": False, "film": False, "poe": False})])
def test_models_forward_backward(name, kw):
    m = build_model(name, **kw)
    pre, post = torch.rand(4, 10, 64, 64), torch.rand(4, 10, 64, 64)
    dem = torch.rand(4, 1, 64, 64) * 500
    region = torch.tensor([0, 1, 2, 3])
    msk = (torch.rand(4, 64, 64) > 0.95).long()
    out = m(pre, post, dem, region)
    assert out["logits"].shape == (4, 1, 64, 64)
    loss, _ = total_loss(out, msk, region, vrex_weight=1.0)
    loss.backward()
    assert torch.isfinite(loss)


def test_dataset_reads_netcdf(tmp_path):
    entries = make_fake_dataset(tmp_path, per_inv=1)
    s = Sen12LS(entries, tmp_path)[0]
    assert s["pre"].shape == (10, 64, 64) and s["post"].shape == (10, 64, 64)
    assert s["dem"].shape == (1, 64, 64) and s["msk"].shape == (64, 64)
    assert s["date_known"]
    nir = 6
    scar = s["msk"].bool()
    drop = (s["pre"][nir] - s["post"][nir])
    assert drop[scar].mean() > 0.1 > drop[~scar].abs().mean()
    assert float(s["pre"].max()) <= 1.0


def test_train_script_end_to_end(tmp_path):
    entries = make_fake_dataset(tmp_path / "data")
    split = {"train": [e for e in entries if e["inventory"] != "newzealand"][:-3],
             "val": [e for e in entries if e["inventory"] != "newzealand"][-3:],
             "test": [e for e in entries if e["inventory"] == "newzealand"]}
    sp = tmp_path / "loco_oceania.json"
    sp.write_text(json.dumps(split))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--split", str(sp),
                        "--data-root", str(tmp_path / "data"), "--epochs", "2", "--batch-size", "4",
                        "--workers", "0", "--max-batches", "2", "--device", "cpu",
                        "--out", str(tmp_path / "runs")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(next((tmp_path / "runs").rglob("results.json")).read_text())
    assert set(res["test"]) == {"all", "newzealand"}
