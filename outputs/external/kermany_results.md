# External validation on the Kermany OCT test set

Models trained on NEH only (no retraining, no tuning on Kermany). Kermany test: 1,000 central B-scans, 250 each NORMAL / DRUSEN / CNV / DME (DME never seen by any model). B-scan ROC-AUC; CI = 1000 patient bootstrap resamples within class. `_mapmax` = max of the Gaussian-smoothed anomaly map (Aresta et al. 2025) instead of the whole-image score. Thresholds = NEH val p95, applied unchanged. Main columns use harmonised images (edge-connected white padding set to black, as NEH's background); the last column uses the raw images.

## Comparison with Aresta et al. 2025 (Kermany test, B-scan ROC-AUC)

| method | all abnormal | drusen | CNV | DME | all abnormal, raw images |
|---|---|---|---|---|---|
| **Aresta 2025 (reverse distillation, trained on 176+ in-house normal volumes)** | 0.87 | 0.70 | 0.95 | 0.95 | |
| ae | 0.580 [0.540, 0.618] | 0.525 | 0.618 | 0.597 | 0.576 |
| svdd_frozen (val-chosen) | 0.827 [0.809, 0.855] | 0.735 | 0.940 | 0.805 | 0.835 |
| svdd_ft (val-chosen) | 0.658 [0.624, 0.699] | 0.590 | 0.700 | 0.683 | 0.635 |
| vae | 0.751 [0.732, 0.794] | 0.505 | 0.914 | 0.836 | 0.620 |
| mkd (val-chosen) | 0.813 [0.794, 0.846] | 0.616 | 0.926 | 0.895 | 0.688 |
| ae_mapmax (val-chosen) | 0.527 [0.505, 0.581] | 0.481 | 0.576 | 0.525 | 0.568 |
| svdd_frozen_mapmax | 0.724 [0.707, 0.774] | 0.581 | 0.826 | 0.765 | 0.655 |
| vae_mapmax (val-chosen) | 0.743 [0.730, 0.796] | 0.577 | 0.873 | 0.780 | 0.553 |
| mkd_mapmax | 0.618 [0.594, 0.675] | 0.551 | 0.658 | 0.644 | 0.564 |

## Same scores on NEH (B-scan ROC-AUC) and threshold transfer to Kermany

| score | NEH val | NEH test | NEH test drusen | NEH test CNV | Kermany PR-AUC (base 0.75) | Kermany specificity @ NEH thr | Kermany sensitivity @ NEH thr |
|---|---|---|---|---|---|---|---|
| ae | 0.636 | 0.570 | 0.489 | 0.701 | 0.790 | 0.800 | 0.285 |
| svdd_frozen | 0.650 | 0.672 | 0.551 | 0.867 | 0.936 | 0.556 | 0.861 |
| svdd_ft | 0.527 | 0.522 | 0.474 | 0.600 | 0.836 | 0.808 | 0.408 |
| vae | 0.712 | 0.674 | 0.541 | 0.890 | 0.891 | 0.940 | 0.425 |
| mkd | 0.684 | 0.629 | 0.502 | 0.834 | 0.926 | 0.956 | 0.439 |
| ae_mapmax | 0.728 | 0.680 | 0.586 | 0.832 | 0.754 | 0.556 | 0.527 |
| svdd_frozen_mapmax | 0.555 | 0.557 | 0.510 | 0.633 | 0.869 | 0.996 | 0.024 |
| vae_mapmax | 0.716 | 0.700 | 0.585 | 0.885 | 0.878 | 0.928 | 0.320 |
| mkd_mapmax | 0.680 | 0.619 | 0.570 | 0.697 | 0.784 | 0.796 | 0.233 |

Validation choice (main score vs map-max, by NEH val B-scan ROC-AUC): ae -> ae_mapmax, svdd_frozen -> svdd_frozen, vae -> vae_mapmax, mkd -> mkd, svdd_ft -> svdd_ft

## NEH test, case level (max over each patient's B-scans)

| score | case ROC-AUC [95% CI] |
|---|---|
| ae | 0.503 [0.358, 0.634] |
| svdd_frozen | 0.814 [0.751, 0.875] |
| svdd_ft | 0.443 [0.340, 0.550] |
| vae | 0.722 [0.637, 0.804] |
| mkd | 0.663 [0.563, 0.760] |
| ae_mapmax | 0.658 [0.513, 0.796] |
| svdd_frozen_mapmax | 0.685 [0.547, 0.812] |
| vae_mapmax | 0.768 [0.656, 0.869] |
| mkd_mapmax | 0.665 [0.510, 0.796] |
