# Validation metrics (mid-review, test set untouched)

Threshold: 95th percentile of val clean-normal scores (B-scan) / val NORMAL patients' scores (patient).

| method | level | n | roc_auc | pr_auc | accuracy | precision | recall | specificity | f1 |
|---|---|---|---|---|---|---|---|---|---|
| M1 Autoencoder | B-scan | 3674 | 0.636 | 0.664 | 0.484 | 0.729 | 0.127 | 0.940 | 0.216 |
| M1 Autoencoder | patient | 98 | 0.618 | 0.858 | 0.204 | 0.750 | 0.037 | 0.944 | 0.071 |
| M2 Deep SVDD (frozen) | B-scan | 3674 | 0.649 | 0.737 | 0.551 | 0.867 | 0.234 | 0.954 | 0.369 |
| M2 Deep SVDD (frozen) | patient | 98 | 0.771 | 0.944 | 0.592 | 0.976 | 0.512 | 0.944 | 0.672 |
| M2 Deep SVDD (fine-tuned) | B-scan | 3674 | 0.588 | 0.639 | 0.467 | 0.697 | 0.086 | 0.952 | 0.153 |
| M2 Deep SVDD (fine-tuned) | patient | 98 | 0.537 | 0.841 | 0.235 | 0.857 | 0.075 | 0.944 | 0.138 |