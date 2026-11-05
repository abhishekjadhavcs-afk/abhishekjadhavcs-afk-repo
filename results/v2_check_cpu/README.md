# v2 CPU check (east_asia, seed 42, 15 epochs × 120 batches; direction only)

| Model | Params | F1@0.5 | AP | F1@val-thr (thr) | Precision | Recall |
|---|---|---|---|---|---|---|
| U-Net | 1.93M | **0.405** | **0.431** | **0.347** (0.66) | **0.495** | 0.343 |
| U-Net wide (capacity-matched) | 2.45M | **0.406** | 0.413 | 0.333 (0.68) | 0.462 | **0.362** |
| PhyDG v2 | 2.46M | 0.352 | 0.400 | 0.316 (0.58) | 0.473 | 0.280 |
| *v1 PhyDG full (for reference)* | 2.12M | 0.344 | 0.351 | 0.296 (0.66) | 0.370 | 0.321 |

- v2 recovers most of v1's deficit on this fold: AP 0.351 → 0.400, and precision back to the baseline level.
- It is still behind both baselines: AP −0.013 vs the wide U-Net and −0.031 vs the U-Net, mainly from lower recall.
- The two baselines differ by 0.018 AP between themselves, which gives a rough sense of run-to-run spread.
  v2's AP gap to the wide U-Net is within that spread; its F1@0.5 gap is not, because it under-predicts at 0.5.
- One fold, one seed: this cannot show whether v2 helps overall. That is what the Kaggle grid is for.
