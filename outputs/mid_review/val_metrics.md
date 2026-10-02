# Validation metrics (mid-review, test set untouched)

Threshold: 95th percentile of val clean-normal scores (B-scan) / val NORMAL patients' scores (patient).

| method | level | n | roc_auc | pr_auc | accuracy | precision | recall | specificity | f1 |
|---|---|---|---|---|---|---|---|---|---|
| M1 Autoencoder | B-scan | 3674 | 0.636 | 0.662 | 0.483 | 0.731 | 0.123 | 0.942 | 0.210 |
| M1 Autoencoder | patient | 98 | 0.615 | 0.854 | 0.194 | 0.667 | 0.025 | 0.944 | 0.048 |
| M2 Deep SVDD (frozen) | B-scan | 3674 | 0.650 | 0.738 | 0.553 | 0.863 | 0.240 | 0.952 | 0.375 |
| M2 Deep SVDD (frozen) | patient | 98 | 0.769 | 0.944 | 0.582 | 0.976 | 0.500 | 0.944 | 0.661 |
| M2 Deep SVDD (fine-tuned) | B-scan | 3674 | 0.527 | 0.603 | 0.466 | 0.702 | 0.080 | 0.957 | 0.144 |
| M2 Deep SVDD (fine-tuned) | patient | 98 | 0.501 | 0.842 | 0.337 | 0.941 | 0.200 | 0.944 | 0.330 |