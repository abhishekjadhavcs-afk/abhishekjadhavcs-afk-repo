# CPU pilot (direction only, not publishable)

Setup: leave-one-cluster-out on Sen12LS S12LS-LD (S2 + DEM), 1 seed (42), 15 epochs × 120
batches of 16 (about 1/2 of a full epoch schedule and 1/4 of the planned 60 epochs), 4 CPU cores.
Checkpoint selected on validation AP from the training regions only.

| Held-out region | Model | F1 | AP | Precision | Recall | IoU |
|---|---|---|---|---|---|---|
| New Zealand (oceania) | U-Net | 0.244 | 0.225 | 0.214 | 0.285 | 0.139 |
| New Zealand (oceania) | PhyDG-Slide | **0.305** | **0.283** | **0.356** | 0.267 | **0.180** |
| Chimanimani (africa) | U-Net | 0.371 | 0.454 | 0.232 | **0.932** | 0.228 |
| Chimanimani (africa) | PhyDG-Slide | **0.437** | **0.481** | **0.294** | 0.848 | **0.280** |
| **Mean** | U-Net | 0.308 | 0.340 | | | |
| **Mean** | PhyDG-Slide | **0.371** | **0.382** | | | |

## Observations
- Large in-region vs unseen-region gap for the baseline: on New Zealand, validation AP was about 0.41 and held-out AP 0.22.
- PhyDG-Slide is better on both folds and on every metric except recall. The gain comes mainly from precision (fewer false alarms).
- On Chimanimani both models over-predict (recall 0.85–0.93, precision 0.23–0.29): the 0.5 threshold is miscalibrated
  for this region. This supports reporting AP and adding a calibration analysis.

## Limits
One seed, two of eight folds, shortened training, no ablations. Run-to-run variance is unknown, so a 0.04–0.06
difference could be partly noise. The full grid is in `notebooks/run_experiments_gpu.ipynb`.
