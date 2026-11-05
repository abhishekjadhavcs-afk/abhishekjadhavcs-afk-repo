# Targeted ablation (CPU, direction only)

Setup: LOCO folds europe and east_asia (where PhyDG lost in the 60-epoch Kaggle runs), seed 42,
15 epochs × 120 batches of 16, the same schedule for every model. Checkpoint chosen on validation AP from the training regions.
"F1@val-thr" uses the threshold that maximised F1 on the validation split (training regions only).

| Fold | Variant | F1@0.5 | AP | F1@val-thr (thr) |
|---|---|---|---|---|
| europe | U-Net | 0.275 | 0.254 | 0.307 (0.69) |
| europe | PhyDG full | 0.240 | 0.306 | 0.294 (0.67) |
| europe | − V-REx | 0.253 | 0.293 | 0.297 (0.67) |
| europe | − style randomisation | **0.288** | 0.314 | **0.332** (0.66) |
| europe | − plausibility map (PoE) | 0.252 | **0.322** | 0.304 (0.67) |
| east_asia | U-Net | **0.405** | **0.431** | **0.347** (0.66) |
| east_asia | PhyDG full | 0.344 | 0.351 | 0.296 (0.66) |
| east_asia | − V-REx | 0.354 | 0.380 | 0.316 (0.64) |
| east_asia | − style randomisation | 0.353 | 0.364 | 0.330 (0.70) |
| east_asia | − plausibility map (PoE) | 0.344 | 0.355 | 0.291 (0.66) |

Mean of the two folds:

| Variant | F1@0.5 | AP | F1@val-thr |
|---|---|---|---|
| U-Net | **0.340** | **0.343** | 0.327 |
| PhyDG full | 0.292 | 0.329 | 0.295 |
| − V-REx | 0.304 | 0.337 | 0.307 |
| − style randomisation | 0.321 | 0.339 | **0.331** |
| − plausibility map | 0.298 | 0.339 | 0.298 |

## Findings
1. Full PhyDG-Slide is worse than U-Net on both folds at the default threshold.
2. Removing any one component helps a little, and removing style randomisation helps most. It is then roughly level with U-Net
   (better on europe, worse on east_asia), not clearly better.
3. On east_asia every PhyDG variant loses on AP as well, so the deficit there is not only calibration. It likely comes from
   something all variants share: change-feature inputs without the raw pre-event image, IBN, or FiLM (not ablated here).
4. Threshold transfer depends on the region. A threshold chosen on the training regions helps on europe (models over-predict there)
   and hurts on east_asia (models under-predict there), for every model. That is a reportable finding about cross-region calibration.

One seed, short schedule: differences under ~0.02 should be treated as noise.
