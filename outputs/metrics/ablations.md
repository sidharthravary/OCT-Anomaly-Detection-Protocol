# Ablations without retraining (test set)

## Case aggregation

| method | aggregation | roc_auc | pr_auc |
|---|---|---|---|
| ae | max | 0.503 | 0.911 |
| ae | mean | 0.501 | 0.912 |
| ae | top5_mean | 0.495 | 0.908 |
| svdd_frozen | max | 0.819 | 0.980 |
| svdd_frozen | mean | 0.766 | 0.972 |
| svdd_frozen | top5_mean | 0.793 | 0.976 |
| svdd_ft | max | 0.547 | 0.917 |
| svdd_ft | mean | 0.614 | 0.941 |
| svdd_ft | top5_mean | 0.521 | 0.912 |
| vae | max | 0.706 | 0.963 |
| vae | mean | 0.685 | 0.958 |
| vae | top5_mean | 0.699 | 0.962 |

## Threshold choice (B-scan level)

The ORACLE row uses test labels and is shown only as an upper reference.

| method | threshold | value | sensitivity | specificity | f1 |
|---|---|---|---|---|---|
| ae | primary (val p95) | 0.001 | 0.107 | 0.923 | 0.186 |
| ae | informed (val Youden) | 0.001 | 0.719 | 0.377 | 0.703 |
| ae | ORACLE (test Youden) | 0.001 | 0.623 | 0.498 | 0.660 |
| svdd_frozen | primary (val p95) | 0.253 | 0.230 | 0.970 | 0.370 |
| svdd_frozen | informed (val Youden) | 0.205 | 0.391 | 0.847 | 0.532 |
| svdd_frozen | ORACLE (test Youden) | 0.212 | 0.370 | 0.879 | 0.516 |
| svdd_ft | primary (val p95) | 0.000 | 0.071 | 0.958 | 0.129 |
| svdd_ft | informed (val Youden) | 0.000 | 0.388 | 0.707 | 0.503 |
| svdd_ft | ORACLE (test Youden) | 0.000 | 0.542 | 0.579 | 0.615 |
| vae | primary (val p95) | 0.006 | 0.247 | 0.939 | 0.386 |
| vae | informed (val Youden) | 0.004 | 0.539 | 0.710 | 0.637 |
| vae | ORACLE (test Youden) | 0.004 | 0.572 | 0.682 | 0.658 |

## Score column

| method | score_column | bscan_roc_auc | case_roc_auc |
|---|---|---|---|
| ae | score | 0.574 | 0.503 |
| ae | score_ssim | 0.529 | 0.442 |
| svdd_frozen | score | 0.672 | 0.819 |
| svdd_ft | score | 0.574 | 0.547 |
| vae | score | 0.668 | 0.706 |
| vae | score_kl | 0.668 | 0.702 |

## File format (B-scan level)

| method | ext | images | bscan_roc_auc | specificity_primary |
|---|---|---|---|---|
| ae | .jpg | 1186 | 0.567 | 0.851 |
| ae | .tif | 5530 | 0.595 | 0.986 |
| svdd_frozen | .jpg | 1186 | 0.741 | 0.994 |
| svdd_frozen | .tif | 5530 | 0.661 | 0.992 |
| svdd_ft | .jpg | 1186 | 0.562 | 0.976 |
| svdd_ft | .tif | 5530 | 0.574 | 0.942 |
| vae | .jpg | 1186 | 0.709 | 0.988 |
| vae | .tif | 5530 | 0.675 | 0.973 |

## Retraining ablations (13_ablation_retrain.py)

`expanded`: normal-labelled B-scans of the ablation_train DRUSEN/CNV patients added to training. `naive`: image-level random split that ignores patients (leakage); its test set is a different random subset of B-scans, so compare the size of the gain, not individual cells. Thresholds as in the main table (primary, from each setting's own val).

| model | setting | level | roc_auc | pr_auc | sensitivity | specificity |
|---|---|---|---|---|---|---|
| svdd_frozen | main (patient split, clean pool) | bscan | 0.672 | 0.815 | 0.230 | 0.970 |
| svdd_frozen | main (patient split, clean pool) | case | 0.819 | 0.980 | 0.476 | 1.000 |
| svdd_frozen | expanded | bscan | 0.671 | 0.814 | 0.228 | 0.971 |
| svdd_frozen | expanded | case | 0.819 | 0.980 | 0.465 | 1.000 |
| svdd_frozen | naive | bscan | 0.669 | 0.804 | 0.276 | 0.936 |
| svdd_frozen | naive | case | 0.876 | 0.931 | 0.580 | 0.933 |

## Excluded layout-outlier patients (never in the main table)

Share of their B-scans above the primary B-scan threshold, and their mean percentile among test scores. High values show the layout shortcut the exclusion avoids.

| method | images | flagged_at_primary | mean_score_percentile_vs_test |
|---|---|---|---|
| ae | 208 | 0.000 | 56.595 |
| svdd_frozen | 208 | 0.995 | 96.561 |
| svdd_ft | 208 | 1.000 | 99.912 |
| vae | 208 | 0.933 | 90.402 |
