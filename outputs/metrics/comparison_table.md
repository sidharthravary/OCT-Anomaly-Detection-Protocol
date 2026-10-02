# Comparison table (test set)

Primary threshold = 95th percentile of val-normal scores; informed threshold = Youden's J on val. Case score = `max` over the patient's B-scans. 95% CIs: 1000 bootstrap resamples of test patients (within class folder). The excluded layout-outlier patients are not included. `pr_auc_baseline` = share of abnormal items, i.e. the PR-AUC of a random scorer (about 0.90 at case level, where 170 of 188 test patients are abnormal).

| method | level | n | roc_auc | roc_auc_ci | pr_auc | pr_auc_ci | pr_auc_baseline | sensitivity_primary | sensitivity_primary_ci | specificity_primary | precision_primary | f1_primary | sens_drusen_primary | sens_cnv_primary | sensitivity_informed | specificity_informed | f1_informed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ae | bscan | 6716 | 0.570 | [0.516, 0.621] | 0.700 | [0.654, 0.750] | 0.656 | 0.106 | [0.075, 0.137] | 0.923 | 0.725 | 0.185 | 0.054 | 0.188 | 0.702 | 0.389 | 0.694 |
| ae | case | 188 | 0.503 | [0.358, 0.634] | 0.911 | [0.877, 0.942] | 0.904 | 0.024 | [0.006, 0.047] | 1.000 | 1.000 | 0.046 | 0.024 | 0.023 | 0.300 | 0.667 | 0.449 |
| svdd_frozen | bscan | 6716 | 0.672 | [0.638, 0.709] | 0.815 | [0.778, 0.847] | 0.656 | 0.235 | [0.204, 0.266] | 0.970 | 0.937 | 0.375 | 0.060 | 0.516 | 0.394 | 0.847 | 0.535 |
| svdd_frozen | case | 188 | 0.814 | [0.751, 0.872] | 0.979 | [0.971, 0.986] | 0.904 | 0.465 | [0.400, 0.524] | 1.000 | 1.000 | 0.635 | 0.190 | 0.733 | 0.629 | 0.944 | 0.770 |
| svdd_ft | bscan | 6716 | 0.522 | [0.492, 0.553] | 0.673 | [0.631, 0.716] | 0.656 | 0.056 | [0.045, 0.068] | 0.952 | 0.690 | 0.104 | 0.031 | 0.097 | 0.145 | 0.864 | 0.238 |
| svdd_ft | case | 188 | 0.443 | [0.339, 0.549] | 0.914 | [0.889, 0.937] | 0.904 | 0.124 | [0.076, 0.176] | 1.000 | 1.000 | 0.220 | 0.071 | 0.174 | 0.135 | 1.000 | 0.238 |
| vae | bscan | 6716 | 0.674 | [0.628, 0.721] | 0.807 | [0.774, 0.842] | 0.656 | 0.259 | [0.225, 0.298] | 0.937 | 0.887 | 0.401 | 0.043 | 0.606 | 0.503 | 0.740 | 0.613 |
| vae | case | 188 | 0.722 | [0.637, 0.801] | 0.965 | [0.953, 0.977] | 0.904 | 0.371 | [0.318, 0.424] | 1.000 | 1.000 | 0.541 | 0.071 | 0.663 | 0.447 | 0.944 | 0.615 |
| mkd | bscan | 6716 | 0.629 | [0.584, 0.679] | 0.773 | [0.730, 0.813] | 0.656 | 0.196 | [0.162, 0.232] | 0.939 | 0.859 | 0.319 | 0.034 | 0.456 | 0.378 | 0.791 | 0.508 |
| mkd | case | 188 | 0.663 | [0.566, 0.757] | 0.956 | [0.942, 0.970] | 0.904 | 0.306 | [0.247, 0.365] | 1.000 | 1.000 | 0.468 | 0.060 | 0.547 | 0.588 | 0.556 | 0.719 |