# PhyDG-Slide

Physics-anchored spectro-temporal domain generalisation for landslide mapping in
**unseen regions**, evaluated on [Sen12Landslides](https://github.com/PaulH97/Sen12Landslides)
(S12LS-LD, Sentinel-2 + DEM, 13 inventories).

## Research question
Can a landslide segmentation model trained on some regions map landslides in a region
it has never seen, without any data from that region? The official S12LS-LD split
is random *within* regions (its README says so), so published scores measure
interpolation, not transfer.

## Protocol: leave-one-cluster-out (LOCO)
`phydg/folds.py` groups the 13 inventories into 8 geographic clusters. Each fold
holds out one cluster entirely as the test set, and validation (for model
selection) comes only from the training regions. No dataset statistics are computed
on test regions: reflectance is scaled physically (DN / 10000) and terrain features
are patch-relative.

| Fold | Held-out inventories | Train | Val | Test |
|---|---|---|---|---|
| europe | italy | 2709 | 477 | 1802 |
| central_asia | kyrgyzstan1, kyrgyzstan2 | 3501 | 616 | 871 |
| east_asia | china, hiroshima, hokkaido | 3790 | 667 | 531 |
| southeast_asia | indonesia, itogon | 3954 | 697 | 337 |
| south_asia | thrissur | 4083 | 719 | 186 |
| africa | chimanimani | 3944 | 695 | 349 |
| caribbean | dominicamaria, usa (Puerto Rico) | 3920 | 691 | 377 |
| oceania | newzealand | 3786 | 667 | 535 |

The split files in `splits/loco/` use the official `splits.json` format, so the
official repo's baselines (U-TAE, U-ConvLSTM, 3D-UNet, ConvGRU) can be trained on
the same folds.

## Model
Input: cloud-masked median composites before and after the event (from the
`pre_post_dates` attribute and SCL), plus the DEM.

| Component | Flag to ablate | Idea |
|---|---|---|
| Spectral style randomisation | `--no-style` | Random smooth band gains, plus cross-region mixing of band statistics, applied identically to pre and post so the change survives and appearance does not |
| Change features | — | post − pre, NDVI/NBR/BSI/NDWI before, after and their difference |
| IBN encoder | `--no-ibn` | Instance norm in shallow layers removes per-patch style |
| Terrain FiLM | `--no-film` | Terrain derivatives (slope, aspect, curvature, roughness, relative elevation) modulate change features at every scale |
| Product of experts | `--no-poe` | final logit = change logit + terrain-plausibility logit |
| V-REx | `--vrex 0` | Variance of per-region risk across regions in a batch; region-balanced sampler guarantees several regions per batch |

Baseline: `--model unet` (early-fused pre, post and terrain).

## Usage
```bash
pip install torch xarray h5netcdf scipy pytest
# download data (needs huggingface.co and us.aws.cdn.hf.co); streams ~39 GB, keeps only the 4,988 LD patches
python scripts/download_s2.py --out data/data_harmonized

python scripts/make_splits.py                       # writes splits/loco/*.json
python scripts/train.py --split splits/loco/loco_oceania.json --data-root data/data_harmonized --model phydg
python scripts/train.py --split splits/loco/loco_oceania.json --data-root data/data_harmonized --model unet
python -m pytest -q tests
```
Results go to `runs/<fold>/<variant>/results.json` (per-region F1, IoU, precision and recall on the landslide class).

## Status
- Region labels confirmed from the official metadata (`inventory` field, and the `<region>_s2_<id>.nc` file names).
- Code is tested end to end on synthetic NetCDF fixtures (`tests/`). **No real-data results yet.**
- Compute: about 2 s per batch of 16 on 4 CPU cores, so about 8 min per epoch. A GPU is needed for the full 8-fold × ablation grid.

## TODO
- Replace `CLUSTERS` with the Sen12Landslides paper's 6-cluster grouping if it can be confirmed, for comparability.
- Verify the `pre_post_dates` handling on real files and report the share of patches with unknown event dates.
- Run the official temporal baselines on the LOCO splits.
- Zero-shot cross-dataset test on Landslide4Sense (post-event frame only).
