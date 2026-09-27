# Comparison table (test set)

Primary threshold = 95th percentile of val-normal scores; informed threshold = Youden's J on val. Case score = `max` over the patient's B-scans. 95% CIs: 1000 bootstrap resamples of test patients (within class folder). The excluded layout-outlier patients are not included. `pr_auc_baseline` = share of abnormal items, i.e. the PR-AUC of a random scorer (about 0.90 at case level, where 170 of 188 test patients are abnormal).

| method | level | n | roc_auc | roc_auc_ci | pr_auc | pr_auc_ci | pr_auc_baseline | sensitivity_primary | sensitivity_primary_ci | specificity_primary | precision_primary | f1_primary | sens_drusen_primary | sens_cnv_primary | sensitivity_informed | specificity_informed | f1_informed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ae | bscan | 6716 | 0.574 | [0.520, 0.624] | 0.701 | [0.656, 0.752] | 0.656 | 0.107 | [0.076, 0.139] | 0.923 | 0.725 | 0.186 | 0.055 | 0.191 | 0.719 | 0.377 | 0.703 |
| ae | case | 188 | 0.503 | [0.358, 0.635] | 0.911 | [0.877, 0.942] | 0.904 | 0.035 | [0.012, 0.065] | 1.000 | 1.000 | 0.068 | 0.024 | 0.047 | 0.329 | 0.667 | 0.483 |
| svdd_frozen | bscan | 6716 | 0.672 | [0.637, 0.709] | 0.815 | [0.778, 0.847] | 0.656 | 0.230 | [0.201, 0.260] | 0.970 | 0.936 | 0.370 | 0.058 | 0.507 | 0.391 | 0.847 | 0.532 |
| svdd_frozen | case | 188 | 0.819 | [0.755, 0.876] | 0.980 | [0.972, 0.987] | 0.904 | 0.476 | [0.418, 0.535] | 1.000 | 1.000 | 0.645 | 0.214 | 0.733 | 0.629 | 0.944 | 0.770 |
| svdd_ft | bscan | 6716 | 0.575 | [0.539, 0.605] | 0.711 | [0.672, 0.752] | 0.656 | 0.070 | [0.057, 0.084] | 0.958 | 0.762 | 0.129 | 0.036 | 0.125 | 0.516 | 0.603 | 0.598 |
| svdd_ft | case | 188 | 0.549 | [0.400, 0.685] | 0.918 | [0.882, 0.952] | 0.904 | 0.071 | [0.035, 0.106] | 0.944 | 0.923 | 0.131 | 0.036 | 0.105 | 0.712 | 0.333 | 0.799 |
| vae | bscan | 6716 | 0.668 | [0.623, 0.717] | 0.802 | [0.768, 0.838] | 0.656 | 0.247 | [0.213, 0.286] | 0.939 | 0.885 | 0.386 | 0.038 | 0.583 | 0.539 | 0.710 | 0.637 |
| vae | case | 188 | 0.706 | [0.619, 0.789] | 0.963 | [0.950, 0.975] | 0.904 | 0.376 | [0.318, 0.435] | 0.944 | 0.985 | 0.545 | 0.083 | 0.663 | 0.747 | 0.444 | 0.827 |
| mkd | bscan | 6716 | 0.628 | [0.581, 0.678] | 0.769 | [0.724, 0.811] | 0.656 | 0.194 | [0.160, 0.230] | 0.932 | 0.846 | 0.315 | 0.038 | 0.444 | 0.398 | 0.764 | 0.523 |
| mkd | case | 188 | 0.641 | [0.539, 0.735] | 0.953 | [0.937, 0.967] | 0.904 | 0.282 | [0.224, 0.341] | 1.000 | 1.000 | 0.440 | 0.060 | 0.500 | 0.271 | 1.000 | 0.426 |