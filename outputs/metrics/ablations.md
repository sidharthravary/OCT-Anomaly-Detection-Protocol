# Ablations without retraining (test set)

## Case aggregation

| method | aggregation | roc_auc | pr_auc |
|---|---|---|---|
| ae | max | 0.503 | 0.911 |
| ae | mean | 0.499 | 0.912 |
| ae | top5_mean | 0.491 | 0.907 |
| svdd_frozen | max | 0.814 | 0.979 |
| svdd_frozen | mean | 0.764 | 0.971 |
| svdd_frozen | top5_mean | 0.793 | 0.976 |
| svdd_ft | max | 0.443 | 0.914 |
| svdd_ft | mean | 0.510 | 0.921 |
| svdd_ft | top5_mean | 0.430 | 0.908 |
| vae | max | 0.722 | 0.965 |
| vae | mean | 0.691 | 0.958 |
| vae | top5_mean | 0.705 | 0.963 |
| mkd | max | 0.663 | 0.956 |
| mkd | mean | 0.625 | 0.948 |
| mkd | top5_mean | 0.648 | 0.953 |

## Threshold choice (B-scan level)

The ORACLE row uses test labels and is shown only as an upper reference.

| method | threshold | value | sensitivity | specificity | f1 |
|---|---|---|---|---|---|
| ae | primary (val p95) | 0.001 | 0.106 | 0.923 | 0.185 |
| ae | informed (val Youden) | 0.001 | 0.702 | 0.389 | 0.694 |
| ae | ORACLE (test Youden) | 0.001 | 0.586 | 0.530 | 0.640 |
| svdd_frozen | primary (val p95) | 0.253 | 0.235 | 0.970 | 0.375 |
| svdd_frozen | informed (val Youden) | 0.205 | 0.394 | 0.847 | 0.535 |
| svdd_frozen | ORACLE (test Youden) | 0.211 | 0.372 | 0.877 | 0.518 |
| svdd_ft | primary (val p95) | 0.000 | 0.056 | 0.952 | 0.104 |
| svdd_ft | informed (val Youden) | 0.000 | 0.145 | 0.864 | 0.238 |
| svdd_ft | ORACLE (test Youden) | 0.000 | 0.503 | 0.537 | 0.576 |
| vae | primary (val p95) | 0.006 | 0.259 | 0.937 | 0.401 |
| vae | informed (val Youden) | 0.004 | 0.503 | 0.740 | 0.613 |
| vae | ORACLE (test Youden) | 0.004 | 0.647 | 0.608 | 0.699 |
| mkd | primary (val p95) | 2.201 | 0.196 | 0.939 | 0.319 |
| mkd | informed (val Youden) | 1.981 | 0.378 | 0.791 | 0.508 |
| mkd | ORACLE (test Youden) | 1.806 | 0.614 | 0.566 | 0.666 |

## Score column

| method | score_column | bscan_roc_auc | case_roc_auc |
|---|---|---|---|
| ae | score | 0.570 | 0.503 |
| ae | score_ssim | 0.528 | 0.444 |
| svdd_frozen | score | 0.672 | 0.814 |
| svdd_ft | score | 0.522 | 0.443 |
| vae | score | 0.674 | 0.722 |
| vae | score_kl | 0.673 | 0.712 |
| mkd | score | 0.629 | 0.663 |

## File format (B-scan level)

| method | ext | images | bscan_roc_auc | specificity_primary |
|---|---|---|---|---|
| ae | .jpg | 1186 | 0.566 | 0.857 |
| ae | .tif | 5530 | 0.591 | 0.986 |
| svdd_frozen | .jpg | 1186 | 0.742 | 0.994 |
| svdd_frozen | .tif | 5530 | 0.662 | 0.992 |
| svdd_ft | .jpg | 1186 | 0.528 | 0.949 |
| svdd_ft | .tif | 5530 | 0.521 | 0.965 |
| vae | .jpg | 1186 | 0.717 | 0.985 |
| vae | .tif | 5530 | 0.681 | 0.967 |
| mkd | .jpg | 1186 | 0.663 | 0.982 |
| mkd | .tif | 5530 | 0.627 | 0.977 |

## Retraining ablations (13_ablation_retrain.py)

`expanded`: normal-labelled B-scans of the ablation_train DRUSEN/CNV patients added to training. `naive`: image-level random split that ignores patients (leakage); its test set is a different random subset of B-scans, so compare the size of the gain, not individual cells. Thresholds as in the main table (primary, from each setting's own val).

| model | setting | level | roc_auc | pr_auc | sensitivity | specificity |
|---|---|---|---|---|---|---|
| ae | main (patient split, clean pool) | bscan | 0.570 | 0.700 | 0.106 | 0.923 |
| ae | main (patient split, clean pool) | case | 0.503 | 0.911 | 0.024 | 1.000 |
| ae | expanded | bscan | 0.565 | 0.696 | 0.097 | 0.926 |
| ae | expanded | case | 0.489 | 0.908 | 0.029 | 1.000 |
| ae | naive | bscan | 0.595 | 0.701 | 0.052 | 0.957 |
| ae | naive | case | 0.648 | 0.747 | 0.092 | 0.933 |
| svdd_frozen | main (patient split, clean pool) | bscan | 0.672 | 0.815 | 0.235 | 0.970 |
| svdd_frozen | main (patient split, clean pool) | case | 0.814 | 0.979 | 0.465 | 1.000 |
| svdd_frozen | expanded | bscan | 0.672 | 0.815 | 0.232 | 0.972 |
| svdd_frozen | expanded | case | 0.815 | 0.979 | 0.465 | 1.000 |
| svdd_frozen | naive | bscan | 0.669 | 0.804 | 0.273 | 0.938 |
| svdd_frozen | naive | case | 0.877 | 0.931 | 0.556 | 0.942 |
| svdd_ft | main (patient split, clean pool) | bscan | 0.522 | 0.673 | 0.056 | 0.952 |
| svdd_ft | main (patient split, clean pool) | case | 0.443 | 0.914 | 0.124 | 1.000 |
| svdd_ft | expanded | bscan | 0.519 | 0.680 | 0.077 | 0.948 |
| svdd_ft | expanded | case | 0.599 | 0.944 | 0.112 | 1.000 |
| svdd_ft | naive | bscan | 0.581 | 0.723 | 0.169 | 0.918 |
| svdd_ft | naive | case | 0.816 | 0.900 | 0.420 | 0.933 |

## Excluded layout-outlier patients (never in the main table)

Share of their B-scans above the primary B-scan threshold, and their mean percentile among test scores. High values show the layout shortcut the exclusion avoids.

| method | images | flagged_at_primary | mean_score_percentile_vs_test |
|---|---|---|---|
| ae | 208 | 0.000 | 58.698 |
| svdd_frozen | 208 | 0.995 | 96.479 |
| svdd_ft | 208 | 1.000 | 99.907 |
| vae | 208 | 0.971 | 91.105 |
| mkd | 208 | 0.817 | 88.872 |
