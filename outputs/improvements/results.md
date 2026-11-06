# Improvements: patch-level detection, ensemble, slice smoothing (one-class)

All models are fitted on clean-normal train B-scans only (no disease images). Every choice was made on NEH validation before reading test. Test = 188 patients / 6,716 B-scans; case = max over a patient's B-scans (95% CI: 1,000 patient bootstraps). Thresholds from val clean-normal scores.

## Choices (validation only)

- PatchCore variant: pc_r18_l12 (val AUC pc_r18_l23 0.759, pc_r18_l12 0.762)
- Ensemble members: ae_mapmax + patchcore (val AUC 0.779; all five: 0.756; 26 subsets tried)
- Slice smoothing for patchcore: window 5 (val AUC w1 0.762, w3 0.782, w5 0.794)
- Slice smoothing for ensemble: window 5 (val AUC w1 0.779, w3 0.797, w5 0.807)

## NEH test

| method | val B-scan AUC | B-scan AUC | PR-AUC (base 0.656) | drusen AUC | CNV AUC | sens / spec @ p95 | case AUC [95% CI] |
|---|---|---|---|---|---|---|---|
| M2 frozen Deep SVDD (previous best) | 0.650 | 0.672 | 0.815 | 0.551 | 0.867 | 0.235 / 0.970 | 0.814 [0.751, 0.875] |
| M3 VAE, map-max | 0.716 | 0.700 | 0.829 | 0.585 | 0.885 | 0.283 / 0.945 | 0.768 [0.656, 0.869] |
| M1 AE, map-max | 0.728 | 0.680 | 0.800 | 0.586 | 0.832 | 0.312 / 0.906 | 0.658 [0.513, 0.796] |
| M4 MKD | 0.684 | 0.629 | 0.773 | 0.502 | 0.834 | 0.196 / 0.939 | 0.663 [0.563, 0.760] |
| M5 PatchCore, layer2+3 | 0.759 | 0.747 | 0.864 | 0.632 | 0.933 | 0.357 / 0.956 | 0.835 [0.768, 0.900] |
| M5 PatchCore, layer1+2 | 0.762 | 0.748 | 0.860 | 0.635 | 0.930 | 0.361 / 0.948 | 0.813 [0.724, 0.893] |
| M5 PatchCore (val-chosen) | 0.762 | 0.748 | 0.860 | 0.635 | 0.930 | 0.361 / 0.948 | 0.813 [0.724, 0.893] |
| Ensemble (val-chosen members) | 0.779 | 0.748 | 0.851 | 0.637 | 0.927 | 0.434 / 0.908 | 0.772 [0.642, 0.875] |
| Ensemble (all five) | 0.756 | 0.730 | 0.846 | 0.606 | 0.931 | 0.388 / 0.924 | 0.827 [0.724, 0.906] |
| M5 PatchCore + slice smoothing | 0.794 | 0.777 | 0.878 | 0.666 | 0.955 | 0.414 / 0.947 | 0.803 [0.718, 0.881] |
| Ensemble + slice smoothing | 0.807 | 0.771 | 0.863 | 0.661 | 0.947 | 0.520 / 0.868 | 0.834 [0.754, 0.904] |

## Versus Sotoudeh-Paima et al. 2022 (same worst-case B-scan subset of our test patients)

Supervised FPN-VGG16 (3-class, trained with disease labels, patient-level 5-fold CV): accuracy 92.0%, sensitivity 91.8%, specificity 95.8% (3-class averages). Ours: normal vs abnormal, threshold = 95.8th percentile of val clean-normal scores.

| method | worst-case AUC | sensitivity | specificity |
|---|---|---|---|
| M2 frozen Deep SVDD (previous best) | 0.706 | 0.250 | 0.994 |
| M3 VAE, map-max | 0.746 | 0.308 | 0.959 |
| M1 AE, map-max | 0.716 | 0.316 | 0.949 |
| M4 MKD | 0.641 | 0.210 | 0.981 |
| M5 PatchCore, layer2+3 | 0.790 | 0.377 | 0.969 |
| M5 PatchCore, layer1+2 | 0.791 | 0.386 | 0.966 |
| M5 PatchCore (val-chosen) | 0.791 | 0.386 | 0.966 |
| Ensemble (val-chosen members) | 0.796 | 0.454 | 0.948 |
| Ensemble (all five) | 0.773 | 0.411 | 0.972 |
| M5 PatchCore + slice smoothing | 0.821 | 0.444 | 0.974 |
| Ensemble + slice smoothing | 0.822 | 0.546 | 0.934 |

## Versus Aresta et al. 2025 (Kermany test, B-scan AUC; no slice smoothing: single B-scans)

| method | all abnormal | drusen | CNV | DME |
|---|---|---|---|---|
| Aresta 2025 | 0.87 | 0.7 | 0.95 | 0.95 |
| M2 frozen Deep SVDD (previous best) | 0.827 | 0.735 | 0.940 | 0.805 |
| M3 VAE, map-max | 0.743 | 0.577 | 0.873 | 0.780 |
| M1 AE, map-max | 0.527 | 0.481 | 0.576 | 0.525 |
| M4 MKD | 0.813 | 0.616 | 0.926 | 0.895 |
| M5 PatchCore, layer2+3 | 0.918 | 0.779 | 0.992 | 0.985 |
| M5 PatchCore, layer1+2 | 0.855 | 0.727 | 0.947 | 0.891 |
| M5 PatchCore (val-chosen) | 0.855 | 0.727 | 0.947 | 0.891 |
| Ensemble (val-chosen members) | 0.705 | 0.618 | 0.778 | 0.717 |
| Ensemble (all five) | 0.781 | 0.664 | 0.868 | 0.813 |
